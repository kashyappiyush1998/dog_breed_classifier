# dog_breed_classifier
This is a web app in which given an image of a dog, algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, it will identify the resembling dog breed.

To Use this - 
1. Clone this locally and download data from https://drive.google.com/open?id=1f-7uHHIwTHKeT7SH22-zbE8u4JqBW1Qz and save it in dog_breed_projects/scripts/data
2. Create a new virtual environment - 'python -m venv venv'
3. Activate virtual environment (Windows) - venv\Scripts\activate
4. To install all required files use command  - 'pip install -r requirements.txt'
5. To run app go inside dog_breed_project folder - 'cd dog_breed_project'
6. Run python run Script - 'python app.py'
7. Follow the insrtructions below.
8. Go to http://127.0.0.1:3001/

Libraries Used are given in requirements.txt file

### Project Definition -

Dogs are such great animals, available in range of shapes, sizes, colors, looks, fur-variations ,and intelligence. Some of them even look so similar that it might 
be difficult for even a human to rell the difference. We have taken that task on our hand that even human have difficulty telling difference that too in visual
inference that has been considered as forte of human intelligence

We are going to use champ of deep learning for visual inference Convolutional Neural Networks (CNN).
Our work flow will be as follows -
1. Input the image
2. Check whether it is image of a human - If yes return the breed of closest looking dog, If no proceed further
3. Check whether image is of a dog - If yes return breed of the dog, Else
4. Return that the image is neither a human or a dog.

### Analysis - 

We have 133 different breeds of dog available. Our data is divided into train, validation and test set, So we don't have to split it later. Images available 
here are of varying sizes, so we will have to resize them before training, validation and testing. 

### Conclusion - 

Accuracy for Test set for my model is: 83.3732% , which is alright considering there are 133 breeds so rando guessing will give us accuracy of: 0.75%
To improve performance we could - 
1. Increase data
2. Choose beeter hyper parametres like varying learning rate
3. Select deeper networks like Resnet152
4. Train model from scratch for better performance.

I will also leave you with some images 

![dog_predict_1](/images/dog_predict_1.png)   
![dog_predict_2](/images/dog_predict_2.png)
![dog_predict_3](/images/dog_predict_3.png)
