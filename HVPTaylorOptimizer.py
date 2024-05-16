import torch
import torch.nn.functional as F
from torch.optim import Optimizer

class HVPTaylorOptimizer(Optimizer):
    def __init__(self, params, lr=1e-2, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False) -> None:
        if lr and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(HVPTaylorOptimizer, self).__init__(params, defaults)

        for group in self.param_groups:
            group.setdefault('nesterov', False) 

            with torch.no_grad():
                for p in group['params']:
                    state = self.state[p]
                    state["current_master"] = p.clone().detach()
                    

        self.iteration = 0     
        self.cur_epoch = -1
        self.buffer_net = None  
        self.num_local_workers = 0  
        self.variance_factor = -1

    ##############################################
    # 시뮬레이션을 위해 필요한 buffer_net을 Optimizer에 등록
    def register_buffer_net(self, buffer_net):
        self.buffer_net = buffer_net

    def register_num_local_workers(self, num_local_workers):
        self.num_local_workers = num_local_workers

    def register_variance_factor(self, variance_factor):
        self.variance_factor = variance_factor

    def set_epoch(self, cur_epoch):
        self.cur_epoch = cur_epoch

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        if self.buffer_net == None:
            print('Please, register buffer_net to Optimizer first!')
            exit(-1)

        if self.num_local_workers == 0:
            print('Please, register num_local_worker to Optimizer first!')
            exit(-1)

        if self.variance_factor < 0:
            print(f"Please, register variance_factor {self.variance_factor} to Optimizer first!")
            exit(-1)

        if self.cur_epoch < 0:
            print(f"Please, set cur_epoch {self.cur_epoch} to Optimizer first!")
            exit(-1)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        

        EN_G = 0      # 정답과 보정값의 유클리디안 거리
        CS_G = 0      # 정답과 보정값의 코사인 거리
        EN_nonhess = 0  # 정답과 ASGD간의 유클리디안 거리
        CS_nonhess = 0  # 정답과 ASGD간의 코사인 거리
        MLPDN = 0   # Master 모델과 Local 모델 파라미터 차이의 Norm
        THPN = 0    # Hessian 보정치의 Norm

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            vector = [] 
            grads = []  
            param = []  

            #############################################
            # HVP 계산
            with torch.no_grad():
                for main_p in group['params']:
                    if main_p.grad is None:
                        continue

                    ############################################
                    # main_param: local worker 파라미터 W(t)
                    # master_p: 원래 master net 파라미터 W(t+τ)
                    # master_p - main_p = W(t+τ) - W(t)
                    master_p = self.state[main_p]["current_master"]
                    vector.append(master_p.add(main_p, alpha = -1))
                    MLPDN += (master_p - main_p).norm()
                    grads.append(main_p.grad)
                    param.append(main_p)
            
            # vector의 크기가 큰 경우 보정하지 않음
            if MLPDN.item() <=1.1:
                hvp = torch.autograd.grad(outputs=grads, inputs=param, grad_outputs=vector)
                hvp_idx = 0
                for main_p in group['params']:
                    THPN += hvp[hvp_idx].norm()
                    hvp_idx+=1


            with torch.no_grad():
                hvp_idx = 0 

                for main_p, (buffer_p_name, buffer_p) in zip(group['params'], self.buffer_net.named_parameters()): # 현재 main_model 파라미터는 Local Model 파라미터이다.
                    if main_p.grad is None:
                        continue


                    # buffer_p로 정답 gradient 계산(SGD).
                    if buffer_p.grad is None:
                        exit(-1)

                    master_p = self.state[main_p]['current_master']
                    
                    #########################################################
                    # Hessian Approximation
                    #########################################################
                    d_p = main_p.grad
                    help = main_p.grad.clone().detach()
                    # vector의 크기가 1.1 이하이며, hessian 보정항의 크기가 일정 이하인 경우 hessian 보정치 사용
                    if MLPDN.item() <=1.1:
                        if THPN.item() <=150:
                            d_p.add_(self.variance_factor *hvp[hvp_idx])

                    if weight_decay != 0:
                        d_p = d_p.add(master_p, alpha=weight_decay)

                    if momentum != 0:
                        param_state = self.state[main_p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                        if nesterov:
                            d_p = d_p.add(buf, alpha=momentum)
                        else:
                            d_p = buf


                    master_p.add_(d_p, alpha=-group['lr'])
                    main_p.copy_(master_p)



                    EN_nonhess += (help - buffer_p.grad).norm()
                    CS_nonhess += F.cosine_similarity(help.view(1, -1), buffer_p.grad.view(1, -1))

                    EN_G += (d_p - buffer_p.grad).norm()
                    CS_G += F.cosine_similarity(d_p.view(1, -1), buffer_p.grad.view(1, -1))


                    
                    hvp_idx += 1



        EN_nonhess_value = EN_nonhess.item()
        CS_nonhess_value = CS_nonhess.item()/62

        EN_value = EN_G.item()
        CS_value = CS_G.item()/62

        MLPDN_value = MLPDN.item()
        if MLPDN.item() <=1.1:
            if THPN.item() <=150:
                THPN_value = THPN.item()
            else:
                THPN_value = THPN.item()
                print(THPN_value)
                THPN_value = 0
        else:
            THPN_value = 0

        if self.cur_epoch < 100:
            if self.iteration == 0:
                print('Iteration, Nonhess Euclidean Norm, Nonhess Cosine Similarity, Hess Euclidean Norm, Hess Cosine Similarity, Master Local Param Difference Norm, Taylor Hessian Part Norm')

            print(f'{self.iteration},{EN_nonhess_value:.3f},{CS_nonhess_value:.3f},{EN_value:.3f},{CS_value:.3f},{MLPDN_value:.3f},{THPN_value:.3f}')

        self.iteration += 1
        return loss                            
        
