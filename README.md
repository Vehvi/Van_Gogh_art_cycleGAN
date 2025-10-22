# **CycleGAN: City -> Van Gogh Style City**

This project implements a cycleGAN model that tries to transform city images to Van Gogh style paintins of the city. The project uses PyTorch and the official cycleGAN implementation as a base (GitHub - junyanz/pytorch-CycleGAN-and-pix2pix: Image-to-Image Translation in PyTorch).

# **Prjoject Structure**


```
├─ datasets/         # Datasets (only test data)
├─ nets.py           # Generator and Discriminator nets
├─ train.py          # Training loop
├─ test.py           # Image transformation with the trained model
├─ testData.py       # Test for data
├─ testNets.py       # Test for nets
├─ requirements.txt 
└─ README.md
```

# **Datasets**
https://www.kaggle.com/datasets/ipythonx/van-gogh-paintings
https://www.kaggle.com/datasets/heonh0/daynight-cityview?select=day

