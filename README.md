
Description: The Spam Detection System is a machine learning project that leverages the Naive Bayes algorithm to classify text messages as either spam or ham (non-spam). The system analyzes textual data, vectorizes it using the TF-IDF method, and provides accurate predictions on the nature of the messages.

Key Features:

    Data Preprocessing: Includes steps for cleaning and preparing the dataset, ensuring that the text is ready for analysis.
    TF-IDF Vectorization: Utilizes the Term Frequency-Inverse Document Frequency (TF-IDF) method to transform the text data into a numerical format for the machine learning model.
    Naive Bayes Classifier: Implements the Naive Bayes algorithm for effective classification based on learned patterns from the training data.
    Model Evaluation: Evaluates the model's performance using metrics like accuracy, precision, and recall.
    User-Friendly Output: Provides clear outputs indicating whether the input message is classified as spam or ham.

Technologies Used:

    Python
    Scikit-Learn (for machine learning)
    Joblib (for model persistence)
    Pandas (for data manipulation)
    Numpy (for numerical operations)
    
Output :-
Analysis of Naive Bayes Classification Report:

    Class 0 (Ham):
        Precision: 0.96
        Recall: 1.00
        F1-score: 0.98
        Interpretation: The model accurately identifies almost all ham messages, with a very high recall (1.00) indicating it rarely misses a ham message.

    Class 1 (Spam):
        Precision: 1.00
        Recall: 0.75
        F1-score: 0.86
        Interpretation: While the model is very precise in labeling spam (i.e., when it predicts spam, it is usually correct), it misses about 25% of spam messages (lower recall of 0.75).

    Overall Accuracy: 0.97
        Interpretation: Naive Bayes achieves high accuracy but could benefit from better recall on spam messages.

Analysis of SVM Classification Report:

    Class 0 (Ham):
        Precision: 0.97
        Recall: 1.00
        F1-score: 0.99
        Interpretation: The SVM model also performs very well on ham messages, with a slightly higher precision than Naive Bayes.

    Class 1 (Spam):
        Precision: 0.99
        Recall: 0.83
        F1-score: 0.90
        Interpretation: The SVM model has slightly better recall on spam messages compared to Naive Bayes, catching 83% of spam messages and achieving a strong F1-score of 0.90.

    Overall Accuracy: 0.98
        Interpretation: SVM achieves a slightly better overall accuracy than Naive Bayes and handles spam messages with higher recall and F1-score.
