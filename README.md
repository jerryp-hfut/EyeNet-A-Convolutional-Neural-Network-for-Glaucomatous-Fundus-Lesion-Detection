# EyeNet: A Convolutional Neural Network for Glaucomatous Fundus Lesion Detection

Glaucoma, an eye disease that causes irreversible vision loss, poses a major threat to human health worldwide. It is estimated that by 2020, more than 11 million people worldwide will be blinded by glaucoma. Given its lack of early symptoms, early diagnosis is critical to prevent vision damage. However, the lack of specialized ophthalmologists, especially in remote areas, limits early screening and treatment of glaucoma. To address this problem, this study proposes **EyeNet**, a convolutional neural network (CNN)-based glaucomatous fundus lesion detection system aimed at assisting physicians in more efficient glaucoma screening.

## Data and Methodology
We utilized the glaucoma fundus lesion image dataset obtained from Kaggle and performed data enhancement via OpenCV to pinpoint the key fundus optic disc regions, which were used as model inputs. The constructed EyeNet model combines multi-layer convolution, pooling, and fully connected layers to extract image features and perform classification.

## Training and Results
The model was trained using the `BCEWithLogitsLoss` loss function and the `Adam` optimizer, and after 140 generations of iterations, EyeNet achieved a high accuracy of **96.59%** on the test set. Visualization analysis further confirmed the model's ability to effectively identify key features of the fundus optic disc.

## Innovation and Impact
The innovation of this study is the application of deep learning techniques to the automated diagnosis of glaucoma, which optimizes the model inputs through data augmentation techniques and significantly improves the training efficiency and diagnostic accuracy. Although the model needs further validation and optimization in practical clinical applications, its potential to improve the early diagnosis rate of glaucoma is obvious, and it is expected to provide effective technical support to alleviate the problem of glaucoma screening in areas with insufficient medical resources.
