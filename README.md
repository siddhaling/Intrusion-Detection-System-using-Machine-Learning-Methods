# Intrusion-Detection-System-using-Machine-Learning-Methods
Intrusion-Detection-System-using-Machine-Learning-Methods
## Carried out this project along with talented students Neetesh Bhati and Puneet Agarwal

The intrusion detection systems are an integral part of modern communication networks. The business environments require a high level of security to safeguard their private data from any unauthorized personnel. The current intrusion detection systems are a step upgrade from the conventional anti-virus software.
Two main categories based on their working. These are:
    • Network Intrusion Detection Systems (NIDS): These systems continuously monitor the network traffic and analyze the packets for a possible rule infringement.
    • Host-based Intrusion Detection Systems (HIDS): These systems monitor the operating system files of an end-user system to detect malicious software that might temper with its normal functioning. 
  
## Model Block Diagram
![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/1.jpg)

The model block diagram gives us a flow of the entire process. We start with the raw data available in the train.csv file and use the pandas library to manage it efficiently. Every dataset needs to be preprocessed before implementing a model so we used the scikit-learn library to normalize, remove outliers, etc. We used the recursive feature elimination wrapped with random forest classifier to extract most influential features. At the next stage, we created two different models to compare their accuracy and results. Finally, we deployed the more accurate model on the test.csv dataset to predict the new classification.

The dataset used for this project is collected from Kaggle by simulating a wide variety of intrusions in a military network environment. 41 features were obtained for each connection row from both the categories. The class variable is either normal or anomalous. Features for intrusion in network.

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/2.jpg)

Data obtained from https://www.kaggle.com/sampadab17/network-intrusion-detection

## Composition of classes

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/3.jpg)

At this stage we analyzed the two key features – the number of failed logins and the superuser attempted. This gives us a rough idea of how access to the superuser is related to the failed logins. In the event of an attack, it is likely that if you are impersonating someone, you will do multiple logins attempts.

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/4.jpg)

## Results

We used the random forest classifier wrapped by the recursive feature elimination to select the top 10 features from the list.

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/5.jpg)

We used the random forest classification (RFC) at the core of RFE to select the top features. The features in the descending order of priority are: 

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/6.jpg)

Here we look at the correlation between the selected features. 

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/7.jpg)

We created two different models – Naïve Bayes and K Nearest Neighbor to compare and evaluate the performance. We did the train test split to see the results of the model. 

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/8.jpg)

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/9.jpg)

We compared the two models to choose the better option for final prediction of the unknown variable in the dataset.

![alt text](https://github.com/siddhaling/Intrusion-Detection-System-using-Machine-Learning-Methods/blob/main/images/10.jpg)


# Further Projects and Contact
www.researchreader.com

https://medium.com/@dr.siddhaling

Dr. Siddhaling Urolagin,\
PhD, Post-Doc, Machine Learning and Data Science Expert,\
Passionate Researcher, Deep Learning, Machine Learning and applications,\
dr.siddhaling@gmail.com
