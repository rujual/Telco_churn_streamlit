# Telco_churn_streamlit
Streamlit UI rendering of the Kaggle Telco Customer Churn Dataset  


![Graphical UI of the Telco Customer Churn Dataset](https://github.com/rujual/Telco_churn_streamlit/blob/master/Screenshot%20from%202020-04-05%2011-20-35.png)
  

Purpose of this mini-project is to give users a live feel of playing with the ML model using Telco Churn's Dataset
This dataset lists the different user-parameters for different users of Telco's Broadband and Phone service, with the dependent variable - Churn flag
The Churn flag defines whether the Customer has discontinued the service of Telco
A Random-Forest Classifier model has been trained for predicting the Churn flag using this data
I have further clustered this data in 100 clusters using K-Means algorithm
The final goal is to find out the data columns to change for each Data Centroid, so that it predicts a previously churned data point is predicted as non-churned by the Random Forest Classifier model, i.e. we find out which parameter is most affecting the group of customers in each cluster, so that we can rectify it to retain that cluster of Customers.  
  

## Features
### Analyse  
  

![Graphical UI of the Telco Customer Churn Dataset](https://github.com/rujual/Telco_churn_streamlit/blob/master/Screenshot%20from%202020-04-05%2011-21-11.png)
  
  
* Raw Data-table
* Cluster-Centroids of Data, after clustering the data into 100 clusters using K-Means Algorithm  

![Graphical UI of the Telco Customer Churn Dataset](https://github.com/rujual/Telco_churn_streamlit/blob/master/Screenshot%20from%202020-04-05%2011-21-18.png)
  
  
### Filter
  * Use any of the Categorical Data Columns to Filter
  * Apply multiple nested filters

  


### Search
  * By Individual Customer number
  * By Cluster Centroid  

  
### Customize Centroid or Singular Record data to see effect on Churn-Prediction
  * Prediction done using Random Forest Classifier
  * Change data-parameters in columns with Highest correlation with Churn result, and view the result immediately
  * Customize both Cluster Centroids or Individual Records  

![Graphical UI of the Telco Customer Churn Dataset](https://github.com/rujual/Telco_churn_streamlit/blob/master/Screenshot%20from%202020-04-05%2011-21-35.png)
  

### Steps to Run  

Clone/Download-
git clone https://github.com/rujual/Telco_churn_streamlit.git  

Install Streamlit Python library-
$ pip install streamlit  

Open terminal and Go into downloaded directory-  

* Run -
$ streamlit run telco_churn_with_clustering.py  

### Play around, get a hands-on feel of the one of the simpler ML models in action!    
