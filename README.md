# face-recognition-new-level-yolo

**Getting Started:**

1. Clone the Repository:
   - Run `git clone https://github.com/shukur-alom/Training-Video-for-AI-Facial-Recognition.git` to get the project files.

2. Install Dependencies:
   - Navigate to the project directory and run `pip3 install -r requirements.txt` to install the required packages.

3. Prepare Training Data:
   - Place your training video clips in the "Input_video" folder. You can use "shukur.mp4" or check the "inp_video" folder for examples.

4. Data Preprocessing:
   - Run `main.py` to preprocess the video data:
     - Set augmentation size from a single image.
     - Define test and validation sizes as prompted.
     - Use `python main.py` to run the script.

5. Model Training:
   - Train the deep learning model using `train.py`.
   - Specify the number of epochs and image size (carefully) when running the script.
   - Use `python train.py` to start the training process.

6. Using the Trained Model:
   - Open `det.py` and set the path to the trained model.
   - Run `det.py` to utilize the trained model for facial recognition.

Congratulations! You've successfully completed all the steps to create and use your facial recognition system.