# NLP-SentimentAnalysis-Doc2Vec

Data Preprocessing
This GitHub repository contains the code for the data preprocessing and sentiment analysis project. The project focuses on preprocessing Twitter data and training classification models to predict the sentiment of the tweets. The repository includes the necessary files and functions to run the project and evaluate the model's performance.

Data Preprocessing Steps
The Twitter data is preprocessed using the following steps:

Load the data using Pandas and extract the sentiment and text columns.
Convert the sentiment column values from 0 and 4 to 0 and 1, respectively.
Replace text-based emojis with corresponding emotions.
Remove URLs and Twitter mentions.
Convert all text to lowercase.
Remove punctuation, numbers, and special characters except for hashtags and alphabets.
Remove stop words.
Remove words with a length of fewer than 3 characters.
The clean_data() function in preprocessing.py implements these preprocessing steps.

Model Training
In this project, two classification models were used to predict the sentiment of the tweets:

Logistic Regression
Random Forest Classifier
The scikit-learn library was used to train the models, and the gensim library was used to train the Doc2Vec model.

The train_doc2vec(cleaned_documents) function takes a list of cleaned_documents and trains a Doc2Vec model on them. Each document is converted into a tagged document using the TaggedDocument class, which assigns a unique tag to each document. The Doc2Vec model is trained using the Doc2Vec class with a vector size of 4, a minimum word count of 4, and 5 epochs. The hyperparameters are set to low values to reduce runtime, which may affect the accuracy. The trained model is saved to disk and returned.

The tokenize_dataset(cleaned_documents, d2v_model) function takes a list of cleaned_documents and a trained Doc2Vec model as input. It tokenizes each document using the word_tokenize function and obtains the vector representation of the document using the infer_vector method of the trained Doc2Vec model. The function returns an array of vectors.

Evaluation and Results
The models were evaluated on the testing set using accuracy, balanced accuracy, and F1 score. The logistic regression model achieved an accuracy of 0.76, a balanced accuracy of 0.50, and an F1 score of 0.43. The random forest model achieved similar results.

Conclusion
In conclusion, this project demonstrates the use of Doc2Vec for sentiment analysis on Twitter data. Although the logistic regression and random forest models achieved reasonable accuracy, there is still room for improvement. Possible next steps include experimenting with different feature representations, optimizing hyperparameters, and using more advanced machine learning techniques. Increasing the hyperparameter values may improve the accuracy of the models. Please refer to the README.txt file for additional information and instructions on running the code.
