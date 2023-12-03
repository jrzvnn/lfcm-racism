import streamlit as st
import pandas as pd
import numpy as np
import streamlit_functions as stfun
from dotenv import load_dotenv
import os
import torch
#from statsmodels.stats.contingency_tables import mcnemar

load_dotenv()


st.set_page_config(page_title="RACEDetect", page_icon=":radio_button:")

# Define the title with emojis using Markdown
title_text = "LFCM: LEVERAGING COMMENT FEATURES IN MULTIMODAL RACIST POST DETECTION"
emojis = "📑📲"

# Combine the title and emojis using Markdown syntax
st.markdown(f"# {title_text} {emojis}")
st.text("🚀 Welcome to RACEDetect web application, a tool for detecting racist posts using")
st.text("the LFCM model. Leverage comment features and early deep fusion to enhance")
st.text("detection accuracy and create a safer and more inclusive online community.")
st.markdown("---")

# Create a combo list in the menu bar for model selection
selected_model = st.sidebar.selectbox("Select a Model", ["FCM", "LFCM"])

# Create an input to upload a CSV file
st.sidebar.title("Upload Data")
uploaded_files = st.sidebar.file_uploader("📂 Upload Data", type=["csv"], accept_multiple_files=True)

# Add some space above the file uploader
st.sidebar.markdown("<br>", unsafe_allow_html=True)

if uploaded_files:
    st.sidebar.button('Compare FCM and LFCM', type='primary')

# a, b, c, d
contingency_table = {'a': 0, 'b': 0, 'c':0, 'd':0}
# precision, recall, f-measure, accuracy
tp = 0
tn = 0
fp = 0
fn = 0

tt_embeddings = stfun.get_embeddings(f"{os.getenv('TEST_EMBEDDINGS')}/tweet_txt.txt")
it_embeddings = stfun.get_embeddings(f"{os.getenv('TEST_EMBEDDINGS')}/image_txt.txt")
c_embeddings = stfun.get_embeddings(f"{os.getenv('TEST_EMBEDDINGS')}/comments.txt")
targets = stfun.load_targets(f"{os.getenv('ROOT_PATH')}/test.csv")

print('\n\n\n\n')
# print(tp, tn, fp, fn)
for data in stfun.get_uploaded_files(uploaded_files):
    img_tensor = np.zeros((3, 299, 299), dtype=np.float32)
    tt_tensor = np.zeros(150, dtype=np.float32)
    it_tensor = np.zeros(150, dtype=np.float32)
    c_tensor = np.zeros(150, dtype=np.float32)

    # convert to tensors
    img_tensor = torch.from_numpy(img_tensor.copy()).requires_grad_()
    tt_tensor = torch.from_numpy(tt_tensor.copy()).requires_grad_()
    it_tensor = torch.from_numpy(it_tensor.copy()).requires_grad_()
    c_tensor = torch.from_numpy(c_tensor.copy()).requires_grad_()

    # tweet image
    if data['image']:
        img_tensor = stfun.image_to_tensor(data['image'])
        
    _selected_model = stfun.get_selected_model(selected_model)

    if data['tweet_id']:
        # tt embedding
        try:
            _tt_embedding = tt_embeddings[data['tweet_id']]
            tt_tensor = stfun.embedding_to_tensor(_tt_embedding)
        except:
            err = 'no tweet text'

        # it embedding
        try:
            _it_embedding = it_embeddings[data['tweet_id']]
            it_tensor = stfun.embedding_to_tensor(_it_embedding)
        except:
            err = 'no img text'

        # c embedding
        try:
            _c_embedding = c_embeddings[data['tweet_id']]
            c_tensor = stfun.embedding_to_tensor(_c_embedding)
        except:
            err = 'no comment'
        
        
        

        # display info
        st.header('', divider=True)

        # tweet file name
        if (data['tweet_id']):
            st.markdown(f"### filename: {data['tweet_id']}.csv")
        
        if (data['tweet_id']):
        # prediction
            fcm_output = stfun.fcm_evaluate(img_tensor, it_tensor, tt_tensor, targets[data['tweet_id']])
            lfcm_output = stfun.lfcm_evaluate(img_tensor, it_tensor, tt_tensor, c_tensor, targets[data['tweet_id']])
            # output = {'pred': '200'}
            # target = int(targets[data['tweet_id']])
            # if (fcm_output['pred'] == target and lfcm_output['pred'] != target):
            #     # contingency_table['a'] += 1
            #         print(data['tweet_id'])
                
            if _selected_model == 'fcm':
                output = fcm_output
            elif _selected_model == 'lfcm':
                output = lfcm_output

            # print('output pred:', output['pred'])    
            if (output['pred'] == 0):
                st.success(f'This tweet is probably not racist.')
            elif (output['pred'] == 1):
                st.error(f'This tweet is probably racist.')


        if (data['tweet_text']):
            st.markdown(f"**Tweet Text**")
            st.markdown(f"{data['tweet_text']}")

        if data['image']:
            st.markdown(f"**Tweet Image**")
            st.image(data['image'])

        # comments
        if (_selected_model == 'lfcm'):
            if data['comments']:
                st.markdown(f"**Tweet Comments**")
                comments = {'': data['comments']}
                comments = pd.DataFrame(comments)
                st.table(comments)
        
        html_string = ('<br /> <br />')
        st.markdown(html_string, unsafe_allow_html=True)


    # fcm_output = stfun.fcm_evaluate(img_tensor, it_tensor, tt_tensor, targets[data['tweet_id']])
    # lfcm_output = stfun.lfcm_evaluate(img_tensor, it_tensor, tt_tensor, c_tensor, targets[data['tweet_id']])
    # target = int(targets[data['tweet_id']])
    # output = None
    # if (fcm_output['pred'] == target and lfcm_output['pred'] == target and target == 1):
    #     # contingency_table['a'] += 1
    #     if (data['tweet_id'] and data['image'] and data['comments']):
    #         print(data['tweet_id'])
        
    #     if _selected_model == 'fcm':
    #         output = fcm_output
    #     elif _selected_model == 'lfcm':
    #         output = lfcm_output

    #     if (output['pred'] == 0):
    #         st.success(f'This tweet is probably not racist.')
    #     elif (output['pred'] == 1):
    #         st.error(f'This tweet is probably racist.')

        # tp += output['tp']
        # tn += output['tn']
        # fp += output['fp']
        # fn += output['fn']
        # target = int(targets[data['tweet_id']])

        # fcm is model 1
        # lfcm is model 2
        # fcm and lfcm right
        # if (fcm_output['pred'] == target and lfcm_output['pred'] == target):
        #     contingency_table['a'] += 1
        # elif (fcm_output['pred'] == target and lfcm_output['pred'] != target):
        #     contingency_table['b'] += 1
        # elif (fcm_output['pred'] != target and lfcm_output['pred'] == target):
        #     contingency_table['c'] += 1
        # elif (fcm_output['pred'] != target and lfcm_output['pred'] != target):
        #     contingency_table['d'] += 1

        # if (output['pred'] == 0):
        #     st.success(f'This tweet is probably not racist.')
        # elif (output['pred'] == 1):
        #     st.error(f'This tweet is probably racist.')

# precision = 0
# recall = 0
# fmeasure = 0
# accuracy = 0

# # precision
# if (tp + fp) > 0:
#     precision = tp / (tp + fp)

# # recall
# if (tp + fn) > 0:
#     recall = tp / (tp + fn)

# # fmeasure
# if (precision + recall) > 0:
#     fmeasure = (2 * precision * recall) / (precision + recall)

# # accuracy
# if (tp + tn + fp + fn) > 0:
#     accuracy = ((tp + tn) / (tp + fn + tn + fp) * 100)

# scores = [precision, recall, fmeasure, accuracy]
# data = {'Metric': ['Precision', 'Recall', 'F-Measure', 'Accuracy'], 'Score': scores}
# df = pd.DataFrame(data)
# st.markdown(f"### {selected_model} Metric Scores")
# table = st.table(df)

# scores = [tp, tn, fp, fn]
# data = {'Metric': ['tp', 'tn', 'fp', 'fn'], 'Score': scores}
# df = pd.DataFrame(data)
# st.markdown(f"### {selected_model} Metric Scores")
# table = st.table(df)

    # print(img_tensor)
    # print(tt_tensor)
    # print(it_tensor)
    # print(c_tensor)


# contigency table for mc nemar's test
# data = [
#     ["", "LFCM Correct", "LFCM Wrong"],
#     ["FCM Correct", contingency_table['a'], contingency_table['b']],
#     ["FCM_Wrong", contingency_table['c'], contingency_table['d']]
# ]
# df = pd.DataFrame(data)
# st.markdown(f"### Mc Nemar's Test ")
# st.table(df)


# if sum(contingency_table.values()) > 0:
#     print(contingency_table)
#     mcnemar_table = [[contingency_table['a'], contingency_table['b']], 
#                     [contingency_table['c'], contingency_table['d']]]

#     mcnemar_result = mcnemar(mcnemar_table, exact=False, correction=False)

#     pvalue = mcnemar_result.pvalue
#     chosen_sig_lvl = 0.05

#     if pvalue < chosen_sig_lvl:
#         st.info(f"Reject null hypothesis (p-value: {pvalue}).")
#     else:
#         st.info(f"Accept null hypothesis (p-value: {pvalue}).")


    
    








