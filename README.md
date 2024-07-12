# GenConViT-plus
Deepfakes are highly realistic, AI-generated videos that can convincingly depict people saying or doing things they never did. These convincing fakes can spread misinformation, manipulate public opinion, and even damage personal and professional reputations. Our solution uses advanced AI and machine learning to detect deepfake videos by analyzing facial expressions, speech patterns, and contextual cues, enhancing accuracy with video preprocessing techniques.

<br>
<br>
<p align="center">
  <img src="https://github.com/rohinikoli10/GenConViT-plus/assets/134802668/c62a32ce-a2dc-4044-833f-505790ca7c0a" alt="Screenshot 2024-01-22 171447">
</p>


<p align="center">GenConViT+ Architecture </p>

<br>


## Models
      Autoencoder (AE) 
      Variational Autoencoder (VAE) 
      ConvNeXt-Swin 
      CNN3D
      
## Installation
Clone the repository

      git clone https://github.com/rohinikoli10/GenConViT-Plus.git

      cd GenConViT-Plus


Install dependencies

      pip install -r requirements.txt

Training

      python train.py -e <num_epochs> -d <data_path> -m <model_type> -p <pretrained_model> -c <use_cnn3d>

Testing

      python test.py -d <data_path> -m <model_type> -p <pretrained_model>

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests. For major changes, please open an issue first to discuss the changes you would like to make.

## License
This project is licensed under the MIT License
      
   
     
  


