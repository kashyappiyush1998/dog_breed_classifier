# dog_breed_classifier
This is a web app in which given an image of a dog, algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, it will identify the resembling dog breed

Libraries Used are given in requirements.txt file

Project Definition -

Dogs are such great animals, available in range of shapes, sizes, colors, looks, fur-variations ,and intelligence. Some of them even look so similar that it might 
be difficult for even a human to rell the difference. We have taken that task on our hand that even human have difficulty telling difference that too in visual
inference that has been considered as forte of human intelligence

We are going to use champ of deep learning for visual inference Convolutional Neural Networks (CNN).
Our work flow will be as follows -
1. Input the image
2. Check whether it is image of a human - If yes return the breed of closest looking dog, If no proceed further
3. Check whether image is of a dog - If yes return breed of the dog, Else
4. Return that the image is neither a human or a dog.

Analysis - 

We have 133 different breeds of dog available. Our data is divided into train, validation and test set, So we don't have to split it later. Images available 
here are of varying sizes, so we will have to resize them before training, validation and testing. 

Conclusion - 

Accuracy for Test set for my model is: 83.3732% , which is alright considering there are 133 breeds so rando guessing will give us accuracy of: 0.75%
To improve performance we could - 
1. Increase data
2. Choose beeter hyper parametres like varying learning rate
3. Select deeper networks like Resnet152
4. Train model from scratch for better performance.

I will also leave you with some images 
