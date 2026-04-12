python main.py --batch_size=48 --prompt_class_len=3 --lr=0.0001 --optim_type=Adam  --dataset=genimage --gpu=1   --lambda_ 0.1

09-16
python main.py --batch_size=48 --prompt_class_len=3 --lr=0.0001 --optim_type=Adam  --dataset=genimage --gpu=0   --lambda_ 0.1
#直接ISD和repository，有IGD的基座

torchrun --nproc_per_node=3 main_ddp.py --batch_size=48 --prompt_class_len=10 --lr=0.0001 --optim_type=Adam  --dataset=genimage   --lambda_ 0.1