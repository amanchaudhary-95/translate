import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
from PIL import Image

img = Image.open('./.streamlit/logo.png')
dev = """:blue[**NLP Application - Machine Translation**]  
    **Developed by** : :blue[**Group 58**]  
    AMAN CHAUDHARY (2022aa05016)  
    ANEESH DAS (2022aa05135)  
    NAVINDRA RAY (2022aa05024)  
    VINODH KUMAR S (2022aa05190)"""

st.set_page_config(page_title='ML Translator', page_icon=img, layout="wide", initial_sidebar_state="collapsed", menu_items={'About': dev})
st.markdown('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown("""<style> header {visibility: hidden;} </style>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner='Loading model for the first time...It may take a while...Please wait...')
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# def stream_text(resp):
#     for word in resp.split():
#         yield word + " "
#         time.sleep(0.1)

def change_lang():
    st.session_state['text'] = None

if 'greeting' not in st.session_state:
    st.session_state['greeting'] = True
    st.toast(dev)

lang_dict = {'English': 'eng_Latn', 'Hindi': 'hin_Deva', 'Maori': 'mri_Latn', 'Nepali': 'npi_Deva'}

im,m, = st.columns([0.05,1])
im.image(img, width=45)
m.subheader("Translation App", anchor=False, divider='rainbow')

c1, c2 = st.columns(2)
in_lang = c1.radio("Select Language", ["Maori", 'Nepali'], horizontal=True, index=1, on_change=change_lang)
out_lang = c2.radio("Translation Language", ["Maori", 'Nepali'], horizontal=True, index=0)
cont1 = c1.form('form', border=True)
cont2 = c2.container(border=True, height=240)
cont2.write(':blue[**Translated Text:**]')
text = cont1.text_area("Enter Text", placeholder="Enter text to translate", label_visibility="collapsed", height=150, key='text')

cont = st.expander("**:green[Demo Sentences:]**", expanded=False)
cont.write("**Nepali** : सोमबारका दिन, स्ट्यानफोर्ड युनिभर्सिटी स्कुल अफ मेडिसिनका वैज्ञानिकहरूले एक नयाँ डायग्नोस्टिक उपकरणको आविष्कारको घोषणा गरे जसले कोषहरूलाई प्रकारका आधारमा क्रमबद्ध गर्न सक्दछः एउटा सानो प्रिन्ट गर्न सकिने चिप जुन मानक ईंकजेट प्रिन्टरहरू प्रयोग गरेर सम्भवतः लगभग एक अमेरिकी सेन्टको लागतमा निर्माण गर्न सकिन्छ।")
cont.write("**Maori** : I te Mane, i kī ake ngā kaipūtaiao nō Stanford University School of Medicine mō te hanganga o tētahi taputapu whakatau e āhei ai te wewete i ngā pūtau ki ana momo: mō te 1 hēneti U.S pea, he rehu-mihini tā iti nei e taea ana te hanga mā ngā mihini tā inkjet noa.")
cont.write("**English (Google Translator)** : On Monday, scientists from the Stanford University School of Medicine announced the creation of a diagnostic tool that can separate cells into their types: for about 1 U.S. cent, a small spray-machine can It is made using ordinary inkjet printing machines.")
# cont.write("**Hindi (Google Translator)** : सोमवार को, स्टैनफोर्ड यूनिवर्सिटी स्कूल ऑफ मेडिसिन के वैज्ञानिकों ने एक डायग्नोस्टिक टूल के निर्माण की घोषणा की जो कोशिकाओं को उनके प्रकारों में अलग कर सकता है: लगभग 1 अमेरिकी प्रतिशत के लिए, एक छोटी स्प्रे-मशीन कैन इसे साधारण इंकजेट प्रिंटिंग मशीनों का उपयोग करके बनाया जाता है।")
cont.divider()
cont.write("**Nepali** : कामदारहरूले प्रायः उनीहरूले गर्ने कुनै पनि निर्णयका लागि उनीहरूका वरिष्ठ अधिकारीहरूको अनुमोदन प्राप्त गर्नुपर्दछ र उनीहरूका निर्देशनहरू बिना कुनै प्रश्न पालन गर्ने अपेक्षा गरिन्छ।")
cont.write("**Maori** : He nui ngā wā me whai whakaaetanga ngā kaimahi i ō rātou rangatira mō ngā whakatau katoa, ā, e hiahiatia ana ka whai rātou i ngā tohutohu o ō rātou rangatira horekau te pātai atu.")
cont.write("**English (Google Translator)** : Employees often have to get the approval of their superiors for any decision they make and are expected to follow their instructions without question.")
# cont.write("**Hindi (Google Translator)** : कर्मचारियों को अक्सर अपने किसी भी निर्णय के लिए अपने वरिष्ठों की मंजूरी लेनी पड़ती है और उनसे बिना किसी सवाल के उनके निर्देशों का पालन करने की अपेक्षा की जाती है।")

if cont1.form_submit_button("Translate"):
    with cont2:
        tokenizer, model = load_model()
        with st.spinner("Translating..."):
            tokenizer.src_lang = lang_dict[in_lang]
            inputs = tokenizer(text=text, return_tensors="pt")
            translated_tokens = model.generate(**inputs, max_length=1500, forced_bos_token_id=tokenizer.lang_code_to_id[lang_dict[out_lang]])
            resp = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
            # cont2.write_stream(stream_text(resp))
            cont2.write(resp)
