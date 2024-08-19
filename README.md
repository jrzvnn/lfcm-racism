This repository features the LFCM model, which improves racist post detection by incorporating comment features alongside visual and textual data, developed during an undergraduate thesis at the Polytechnic University of the Philippines.

### Project Structure

```
src
├── training
├── preprocessing
data
models
requirement.txt
notebooks
```
### Description

This project is organized into the following directories:

**src:** Contains the source code for the project, including training and preprocessing scripts.

**data:** Contains the data used to train and evaluate the project.

**models:** Contains the trained models.

**notebooks:** Contains Jupyter notebooks for data exploration and analysis.

**requirement.txt:** Contains the list of Python packages required to run the project.

### Usage

To run the project, you will need to install the Python packages listed in requirement.txt. You can do this using the following command:
```
pip install -r requirement.txt
```

### Environment Variables
Before running the project, you need to create a .env file in the root directory of the project. This file should include the following environment variables:
```
ROOT_PATH=
IMAGES_RESIZED_PATH=
TEST_EMBEDDINGS=
FCM_TH=
LFCM_TH=
MODELS_PATH=
```

### Contact
Feel free to reach out through [LinkedIn](https://www.linkedin.com/in/jrz-vnn/) if you have any questions or need further information.
