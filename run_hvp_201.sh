# hvp           : Hessian vector product 방법

epoches=120
for variance_factor in 0.3
do
    for seed in 2015 2016
    do
        for optim in "hvp"
        do
            for batch_size in 256
            do
                for lr in 0.01
                do
                    for num_local_worker in 4
                    do
                        DIRECTORY=./${seed}/log_ResNet18_${optim}_${variance_factor}_${batch_size}_${lr}_${num_local_worker}
                        if [ -d "${DIRECTORY}" ]; then
                            echo "${DIRECTORY} exists!"
                        else
                            mkdir ${DIRECTORY}
                            echo "${DIRECTORY} is created!"
                        fi
                        
                        python3 -u ./taylor_simulation_Gpu0.py --lr ${lr} --batch-size ${batch_size} --num-local-workers ${num_local_worker} --epoches ${epoches} --gpu-id 0 --seed ${seed} --optim ${optim} --variance-factor ${variance_factor} > ${DIRECTORY}/log.txt
                        
                        sleep 3s
                    done
                done
            done
        done
    done
done