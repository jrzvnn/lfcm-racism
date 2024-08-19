import csv
import pandas as pd
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import customTransform
import torch
import lfcm

load_dotenv()

models_path = os.getenv('MODELS_PATH')

def get_data_from_csv(csv, **kwargs):
    data = pd.read_csv(csv)
    tweet_text = ""
    image = None
    comments = []
    tweet_id = ""

    # print(csv)

    try:
        tweet_id = csv.name.replace('.csv', '')
    except:
        tweet_id = ""

    # get tweet text
    try:
        tweet_text = data["Tweet Text"].iloc[0]
    except:
        tweet_text = ""

    # get image url (filepath)
    try:
        image = f"{os.getenv('IMAGES_RESIZED_PATH')}/{tweet_id}.jpg"
        image = Image.open(image)
    except:
        image = None

    try:
        _comments = data["Comments"].iloc[:]
        for c in _comments:
            comments.append(c)
    except:
        comments = []

    return {"tweet_text": tweet_text, "image": image, "comments": comments, 'tweet_id':tweet_id}

def get_uploaded_files(files):
    render_data = []
    for _csv in files:
        data = get_data_from_csv(_csv)
        render_data.append(data)
    return render_data
        # print(data)
    # print('\n\n\n')

def get_selected_model(model):
    selected_model = ''
    if model == "FCM":
        selected_model = 'fcm'
    elif model == "LFCM":
        selected_model = 'lfcm'
    return selected_model
    # print(selected_model)

def get_embeddings(filepath):
    embeddings = {}
    for i,line in enumerate(open(filepath)):
        data = line.strip().split(',')
        tweet_id = data[0]
        embedding = data[1:]
        embeddings[tweet_id] = embedding

    return embeddings


def image_to_tensor(_image=None):
    image = np.zeros((3, 299, 299), dtype=np.float32)
    try:
        image = customTransform.rescale(_image, 299)
        image = customTransform.preprocess_image_to_np_arr(image)
    except Exception as e:
        image = np.zeros((3, 299, 299), dtype=np.float32).requires_grad_()

    return torch.from_numpy(image.copy())

def embedding_to_tensor(str_embedding):
    hidden_state = 150
    embedding = np.zeros(hidden_state, dtype=np.float32)
    try:
        embedding = np.array(str_embedding, dtype=np.float32)
    except:
        embedding = np.zeros(hidden_state, dtype=np.float32)
    # Removed the `return` statement from the `finally` block
    tensor = torch.from_numpy(embedding.copy()).requires_grad_()
    return tensor

def load_model(model_type, checkpoint_filepath):
    if model_type == 'fcm':
        model = lfcm.OldModel()
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        model = torch.nn.DataParallel(model, device_ids=None)
        model = model.to('cpu')
        model.load_state_dict(checkpoint)
        return model
    elif model_type == 'lfcm':
        model = lfcm.LFCM()
        checkpoint = torch.load(checkpoint_filepath, map_location=torch.device('cpu'))
        model = torch.nn.DataParallel(model, device_ids=None)
        model = model.to('cpu')
        model.load_state_dict(checkpoint)
        return model

def load_targets(filepath):
    targets = {}
    with open(filepath, "r", encoding="utf-8") as file:
        csv_reader = csv.reader(file)
        data = [row for row in csv_reader]
        #  remove header
        if len(data) > 0:
            data = data[1:]

        for row in data:
            targets[row[0]] = row[-2]

    return targets


fcm_model = load_model('fcm', f"{models_path}/fcm_e16b24_full.pth")

lfcm_model = load_model('lfcm', f"{models_path}/lfcm_e16b24_full.pth")

def fcm_evaluate(i_ten, it_ten, tt_ten, target):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pred = 0 # 0 or 1
    threshold = float(os.getenv('FCM_TH'))
    target = int(target)

    with torch.no_grad():
        fcm_model.eval()

        output = fcm_model(i_ten.unsqueeze(0), it_ten.unsqueeze(0), tt_ten.unsqueeze(0))

    prediction_weights = output.cpu().numpy().tolist()[0]
    racist_score = prediction_weights[1]
    not_racist_score = prediction_weights[0]
    softmax_racist_score = np.exp(racist_score) / (np.exp(racist_score) + np.exp(not_racist_score))

    if softmax_racist_score >= threshold:
        pred = 1
    else:
        pred = 0

    # get tp, tn, fp, and fn
    if target == 1 and pred == 1:
        tp += 1
    elif target == 0 and pred == 0:
        tn += 1
    elif target == 1 and pred == 0:
        fn += 1
    elif target == 0 and pred == 1:
        fp += 1

    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'pred':pred}


def lfcm_evaluate(i_ten, it_ten, tt_ten, c_ten, target):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    pred = 0 # 0 or 1
    threshold = float(os.getenv('LFCM_TH'))
    target = int(target)

    with torch.no_grad():
        lfcm_model.eval()

        output = lfcm_model(i_ten.unsqueeze(0), it_ten.unsqueeze(0), tt_ten.unsqueeze(0), c_ten.unsqueeze(0))

    prediction_weights = output.cpu().numpy().tolist()[0]
    racist_score = prediction_weights[1]
    not_racist_score = prediction_weights[0]
    softmax_racist_score = np.exp(racist_score) / (np.exp(racist_score) + np.exp(not_racist_score))

    # print(softmax_racist_score)
    if softmax_racist_score >= threshold:
        pred = 1
    else:
        pred = 0

    # get tp, tn, fp, and fn
    if target == 1 and pred == 1:
        tp += 1
    elif target == 0 and pred == 0:
        tn += 1
    elif target == 1 and pred == 0:
        fn += 1
    elif target == 0 and pred == 1:
        fp += 1

    return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'pred':pred}

# def evaluate(model_type, i_ten, it_ten, tt_ten, c_ten, target):
#     tp = 0
#     tn = 0
#     fp = 0
#     fn = 0
#     pred = 0 # 0 or 1
#     threshold = 0
#     model = fcm_model
#     output = None

#     if model_type == 'fcm':
#         threshold = float(os.getenv('FCM_TH'))
#     elif model_type == 'lfcm':
#         threshold == float(os.getenv('LFCM_TH'))
#         model = lfcm_model

#     with torch.no_grad():
#         model.eval()
#         if model_type == 'fcm':
#             output = model(i_ten.unsqueeze(0), it_ten.unsqueeze(0), tt_ten.unsqueeze(0))
#         elif model_type == 'lfcm':
#             output = model(i_ten.unsqueeze(0), it_ten.unsqueeze(0), tt_ten.unsqueeze(0), c_ten.unsqueeze(0))

#     if output:
#         prediction_weights = output.cpu().numpy().tolist()[0]
#         racist_score = prediction_weights[1]
#         not_racist_score = prediction_weights[0]
#         softmax_racist_score = np.exp(racist_score) / (np.exp(racist_score) + np.exp(not_racist_score))

#         if softmax_racist_score >= threshold:
#             pred = 11
#         else:
#             pred = 10

#     return {'tp':tp, 'tn':tn, 'fp':fp, 'fn':fn, 'pred':pred}
