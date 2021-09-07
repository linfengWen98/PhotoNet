# PhotoNet
Photorealistic style transfer

PhotoNet is proposed by "Fast Universal Style Transfer for Artistic and Photorealistic Rendering".

PhotoNAS, a pruned network of PhotoNet, is proposed by "Ultrafast Photorealistic Style Transfer via Neural Architecture Search".

pretrained models [here](https://drive.google.com/drive/folders/1HcwaTBBcooB36uWyEkqrTH7gZ3DyV1VP?usp=sharing).

* train the model
```bash
python train.py --dataset 'dir to images'
```

* image reconstruction and compare
```bash
python reconstruction.py --image ./content/1.jpg
```

* style transfer
```bash
python transfer.py --content ./content/1.jpg --style ./style/1.jpg --output ./out/1.jpg
```
