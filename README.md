# Automatic detection of drug alergies

This project consist in the training of diferent algorithms using anonymous clinic notes. The main point is to establish a relation betweeen these clinic notes and the outcome of patients being alergic or not to some drug. 

---

## Algorithms used

-   **Random Forest**
-   **Gradient Boosting**
-   **Support Vector Machine**

---
## Neural network based models
- **Convolutional neural network**
- **roberta-base-bne** based on RoBERTa, pre-trained using the largest Spanish corpus known to date, with a total of 570GB of clean and deduplicated text processed

---

## Metrics
In order to measure how accurate the models were working I used 3 metrics:
![image](https://github.com/user-attachments/assets/534a3426-85bc-43b5-9669-ee421b38c6f8)

-   **Precision**: Proportion of true positive divided by all the positives guessed by the model.
-   **Recall**: It reflects the capacity of the model to deliver true positives, dividing true positive by (true positives + false negatives).
-   **F1 measure**: This metric uses both of the previous metrics, recall and precision to deliver a measure that portraits the fiability of the model.

---

## Technologies Used

-   **Preprocess of clinic notes**:Spacy
-   **Classic algorithm and metrics**: sklearn
-   **Neural network**: Keras
-   **RoBERTa pretrained model**: Hugging face transformers library
-   Also highligh the use of Numpy to be able to manage the vectors that served as input for the models.
---

## Hardware Used
For the training of the models we were using an i7 8750H, 22 GB RAM and NVIDIA 1050 GPU 

---

## System layout

![image](https://github.com/user-attachments/assets/d3a9e75f-a968-4269-b527-bba9444be825)

In a final release of the whole system, we would only use one model. The one with the best results, here we get access to almost every model trained. Those with results under 0.60<F1 are not included in the final architecture.
