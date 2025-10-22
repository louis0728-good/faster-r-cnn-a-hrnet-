### check_structure.py 我放置在 "D:\rcnn\A-HRNet\data\coco\annotations\check_structure.py" (目前可先不實作這步)
### double_check.py 我放置在 "D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\output_videos\double_check.py"
### video_inference.inynb 我放置在 "D:\rcnn\PyTorch-Object-Detection-Faster-RCNN-Tutorial\video_inference.ipynb"
> clone 專案連結: https://github.com/johschmidt42/PyTorch-Object-Detection-Faster-RCNN-Tutorial
> 一樣按照原作者說的流程安裝必要的套件或東西。
* 要按照你們本機端修改的程式路徑有: model_weights_path, project_dir
* 仔細看我上面附的，有一個檔案是甚麼 osnet..... 那個要下載並且放到相對路徑: PyTorch-Object-Detection-Faster-RCNN-Tutorial\weights\
  
### 2d_detection.py 我放在 "D:\rcnn\deep-high-resolution-net.pytorch\demo\2d_detection.py"
> 2d_detection.py 那個是我目前用別的 2d 檢測模型去跑的，你們要先 clone 到本機端，網址是 https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/tree/master
> 流程就按照作者上面所說的就好，指令則是照我的 python demo/2d_detection.py --cfg demo/inference-config.yaml
* 要按照你們本機的放置路徑而修正的程式碼有: weights_path, input_dir, bbox_base, output_dir
