## Training :

#MobileNetv2
CUDA_VISIBLE_DEVICES=1 python main.py --save Mobilev2 --model mobilev2
#CUDA_VISIBLE_DEVICES=1 python main.py --save Mobilev2_se --attention se --model mobilev2
#CUDA_VISIBLE_DEVICES=1 python main.py --save Mobilev2_cbam --attention cbam --model mobilev2
#CUDA_VISIBLE_DEVICES=1 python main.py --save Mobilev2_ge --attention ge --model mobilev2

#Mnas
#CUDA_VISIBLE_DEVICES=0 python main.py --save mnas --model mnas
#CUDA_VISIBLE_DEVICES=0 python main.py --save mnas_se --attention se --model mnas
#CUDA_VISIBLE_DEVICES=0 python main.py --save mnas_cbam --attention cbam --model mnas
#CUDA_VISIBLE_DEVICES=0 python main.py --save mnas_ge --attention ge --model mnas


#Test : 
CUDA_VISIBLE_DEVICES=1 python main.py --test_only --pre_train experiments/Mobilev2/valid_model.t7 --save test



#Grad CAM:
CUDA_VISIBLE_DEVICES=1 python main.py --gradcam --pre_train experiments/Mobilev2/valid_model.t7

