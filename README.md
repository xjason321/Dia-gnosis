# dia-gnosis
A machine learning program developed to accurately diagnose Type II Diabetes from information.

This program is designed for use by health care professionals and systems.

To use the program, scroll down on the website and enter patient information. There are two methods of input.

1. For individual patients, you may input using the form.
    - Make sure to read each description to fully understand the information requested.  
2. For mass diagnosis, you may upload a csv file, making sure that the headers are as follows:
    - Pregnancies,	Glucose,	BloodPressure,	SkinThickness,	Insulin,	BMI,	DiabetesPedigreeFunction,	Age
    - Two headers will be added onto the uploaded CSV: "Diabetes Prediction", and "Diabetes Prediction Probability"
        - Diabetes Prediction: 0 or 1 indicates Diabetes Negative or Positive.
        - Diabetes Prediction Probability: Number percent indicates % probability of Diabetes.


****Dia-Detect and its creators are not liable for any false medical advice given to its users. 
They are clearly stated to be mere predictions which can be followed up on.****
