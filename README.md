# Simple improvement of yolo-v8  
## This project extends some different use of yolo-v8  
# 1.Environment deployment  
## First install python>=3.10.2  
## On a device with nvidia services, run the following command line to see the cuda version  
`<nvidia-smi>`  
## Install using the following command (X of cu11X should be modified according to cuda version)  
`<pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118>`  
## On a device without nvidia services, install using the following command  
`<pip3 install torch torchvision torchaudio>`  
# 2.Dataset Production
## The voc datasets should be placed in data/xml_data with the following structure  
--simple_YOLO  
    --data  
        --xml_data  
            --Annotations  
                1.xml  
                2.xml  
                ...  
            --Images  
                1.jpg  
                2.jpg  
                ...  
# 3.Adding blocks  
## Add blocks to my_block.py and modify parse_model function in ultralytics/nn/tasks accordingly  
## Now that the yaml model structure has been added and set up, we can run the tests:
`<python model_test.py --model models/c3.yaml>`  
# 4.Training
## After the dataset made as Step 2 has been saved to data/xml_data, run the following command to perform label conversion:  
`<python voc2txt.py --ini_path data/xml_data --out_path data/datasets --file mytrain>`  
## To split dataset  
`<python divide.py --val_rate 0.2 --data data/datasets>`  
## Run training
`<python train.py --file mytrain --model models/yolov8n.yaml --epochs 50 --batch 10 --imgsz 640>`  
## To enable mixed-precision amp training, run the following command (some versions of cuda will prevent the training process from being recorded, so turn off amp)  
`<python train.py --file mytrain --model models/yolov8n.yaml --epochs 50 --batch 10 --imgsz 640 --amp True>`  
# 5.Inference
## Put the image you want to detect into your inference file and run the following command to detect it  
`<python inference.py --file mytrain --train_file train --source inference --conf 0.3>`  


