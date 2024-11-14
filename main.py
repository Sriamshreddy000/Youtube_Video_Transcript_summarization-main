from transcription import fetch_transcript,speech_to_text
from summarize import get_summary,clean_summary
from features import get_vid_data,fetch_translated_text,ttspeech

global transcript
transcript=""

def get_transcript(video_link):
    global transcript
    transcript=""
    try:
        transcript= fetch_transcript(video_link).capitalize()
        transcript=clean_summary(transcript)        
        return transcript
    except Exception as e:
        return f"An error during transcription."

def translate_summary(text,lang_choice):
    try :
        return fetch_translated_text(text,lang_choice).capitalize()
        
    except Exception as e:
       return f"An error during translation."
    
def get_data(link):
    try :
        data=get_vid_data(link)
        title=data['Title']
        duration=data['Duration']
        description=data['Description']
    except Exception as e:
        return f"An error occured during fetching video data."
    
def audio_to_text(link="",audio_file=True):
    try :
        transcript=clean_summary(speech_to_text("",True))
        return transcript
    except Exception as e:
        return f"An error transcribing audio file."
    
def text_to_speech(text,language):
    try :
        tts=ttspeech(text,language)
    except Exception as e:
        return f"An error occured during generating audible summary."

def summarize_transcript(text,model_choice):
    if len(text)<=150:
        return clean_summary(text) 

    from transcription import manual_subtitles
    try :
        summary=get_summary(manual_subtitles,text,model_choice)
    except Exception as e:
        return f"An error occured during summarization."
    return summary    
 
   
# print(summarize_transcript(get_transcript("https://www.youtube.com/watch?v=MrF0mWZQO6o"),1))
# print(summarize_transcript(get_transcript("https://www.youtube.com/watch?v=tR1ECf4sEpw"),2))

