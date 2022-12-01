from surprise import Reader
from surprise import Dataset
import pandas as pd
import numpy as np
from surprise import SVD
from surprise.model_selection import GridSearchCV
from surprise.accuracy import rmse
from surprise.model_selection import KFold
from sklearn.model_selection import train_test_split

def train_SVD_wGridSearch(processed_data_file="/content/drive/MyDrive/Collab Codes/MLIP/RemappedData.csv"):
  
  
  FullDataset=pd.read_csv(processed_data_file)
  
  
  #Our train test validate has been split into 60-20-20 for the sake of experimentation, but we will use full dataset to finally fit the selected model 
  train,testval = train_test_split(FullDataset, test_size=0.4, random_state=42,shuffle=False)
  validate,test = train_test_split(testval, test_size=0.5, random_state=42,shuffle=False)
  print('THE DATA HAS BEEN SPLIT INTO :')
  print("TRAIN SHAPE",train.shape)
  print("VALIDATE SHAPE",validate.shape)
  print("TEST SHAPE",test.shape)

  
  reader = Reader(rating_scale=(1, 5))

  # Loads Pandas dataframe into surprise datasets
  data = Dataset.load_from_df(FullDataset[["user", "item", "rating"]], reader)

  #sperately doing for train test split
  traindata = Dataset.load_from_df(train[["user", "item", "rating"]], reader)
  validatedata = Dataset.load_from_df(validate[["user", "item", "rating"]], reader)
  testdata = Dataset.load_from_df(test[["user", "item", "rating"]], reader)

  
  # Code below was inspired from MLIP recitation

  """
  lr_all is the learning rate for all parameters (how much the parameters are adjusted in each iteration)
  reg_all is the regularization term for all parameters, which is a penalty term added to prevent overfitting.
  """

  # The parameter grid that was utilized for gridsearch
  # param_grid = {
  #     "n_epochs": [5, 10],
  #     "lr_all": [0.002, 0.005],
  #     "reg_all": [0.4, 0.6]
  # }

  # Get the best params using GridSearchCV
  # gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3)
  # gs.fit(data)


  # Post Grid Search we have the below params
  # best_params=best_params = gs.best_params["rmse"]
  best_params={'n_epochs': 10, 'lr_all': 0.005, 'reg_all': 0.4}
  
  # print("BEST PARAM:",best_params)
  # Extract and train model with best params
  svd_algo = SVD(n_epochs=best_params['n_epochs'],
                lr_all=best_params['lr_all'],
                reg_all=best_params['reg_all'])
  
  # Train
  trainingSet = data.build_full_trainset()


  svd_algo.fit(trainingSet)

  # Kfold to also check the test errors as we train the model
  kf = KFold(n_splits=3)
  # errors_tr=[]
  errors_te=[]
  for trainset, testset in kf.split(data):
    svd_algo.fit(trainset)                             
    # predictions_train = svd_algo.test(trainset)
    predictions_test = svd_algo.test(testset)
    # errors_tr.append(rmse(predictions_train))
    errors_te.append(rmse(predictions_test))

  # return svd_algo,(np.mean(errors_tr),np.mean(errors_te))
  return svd_algo,np.mean(errors_te)
