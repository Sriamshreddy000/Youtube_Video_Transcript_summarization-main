import nltk 
nltk.download('punkt')
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import unicodedata
import warnings 
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def get_summary(manual_subtitles, text, model_choice):
    """
    Get the summary of the given text using extractive and/or abstractive summarization.

    Returns:
        str or list: The generated summary. If model_choice is not 1 or 2, a list containing both T5 and DistilBART summaries.
    """
    if manual_subtitles:
        extractive_summary = get_extractive_summary(text)
        abstractive_summary = get_abstractive_summary(extractive_summary, model_choice)
    else:
        abstractive_summary = get_abstractive_summary(text, model_choice)
    
    return abstractive_summary


def get_abstractive_summary(text, model_choice):
    """
    Generate an abstractive summary of the given text using a transformer-based model.
    
    Args:
        text (str): The input text to summarize.
        model_choice (int): The choice of model:
            - 1: T5-base model
            - 2: DistilBART-CNN-12-6 model
    
    Returns:
        str or list: The generated summary. If model_choice is not 1 or 2, a list containing both T5 and DistilBART summaries.
    """
    # Select the appropriate model pipeline based on the model_choice
    if model_choice == 1:
        generator = pipeline('summarization', model='t5-base')
    elif model_choice == 2:
        generator = pipeline('summarization', model='sshleifer/distilbart-cnn-12-6')
    else:
        # If model_choice is not 1 or 2, recursively call the function for both models and return a list of summaries
        t5_summary = get_abstractive_summary(text, model_choice=1)
         
        distilbart_summary = get_abstractive_summary(text, model_choice=2)
        return [t5_summary, distilbart_summary]
        

    full_summary = ""
    start, chunk_size = 0, 1200
    end = min(chunk_size, len(text))
    
    # Determine the length parameter based on the length of the text
    if len(text) < 2000:
        length = 80
    elif len(text) > 20000:
        length = 60
    else:
        length = 70
    
    # Generate the summary in chunks until the full text is processed
    while start < len(text):
        chunk = text[start:end]
        summary = generator(chunk, min_length=length, do_sample=False)
        full_summary += summary[0]["summary_text"]
        start = end
        end = min(start + chunk_size, len(text))
       
    full_summary=clean_summary(full_summary)
    del generator

    return full_summary


def get_extractive_summary(text):
    """
    Generate an extractive summary of the given text using TextRank algorithm.
    
    Args:
        text (str): The input text to summarize.
    
    Returns:
        int: The length of the extractive summary.
    """
    
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    req_sentences = round(len(sent_tokenize(text)) * 0.70)
    summarizer = TextRankSummarizer()
    summary = summarizer(parser.document, req_sentences)
    ext_summary = ""

    for sentence in summary:
        ext_summary += str(sentence)

    return ext_summary

def clean_summary(text):
    error_dict=["An error during transcription.","An error during translation.","An error occured during fetching video data.","An error transcribing audio file.","An error occured during generating audible summary.","An error occured during summarization.","Video not found, enter a valid youtube video link.","An error occured during transcription.",'An Error occurred with given link.',"Only english language is supported for transcription."]
    for i in error_dict:
        if i==text:
            return text
    irrelevant_terms = ["[music]", "[Music]", "\n","<<",">>"]
    sentence_list = sent_tokenize(text)
    
    # for term in irrelevant_terms:
    #     relevant_text = [sentence.replace(term, "").strip() for sentence in sentence_list]
    for i in range(len(sentence_list)):
        for item in irrelevant_terms:
            sentence_list[i]=sentence_list[i].replace(item,"")

    s=""
    for item in sentence_list:
        item=item.strip()
        s=s+" "+item.capitalize()
   
    normalized_text = unicodedata.normalize('NFKD', s)
    formatted_text = normalized_text.encode('ascii', 'ignore').decode('ascii')
    cleaned_text = formatted_text.replace("\'", "'")
    return cleaned_text

# print(get_summary(True,"""When we last left nvidia, the company had emerged victorious in the brutal graphics card battle royale throughout the 1990s. very impressive. but as the company entered the 2000s, they embarked on a journey to do more. moving towards an entirely new kind of microprocessor - and the multi-billion dollar market it would unlock. in this video, we are going to look at how nvidia turned the humble graphics card into a platform that dominates one of tech’s most important fields: artificial intelligence. in 1999, nvidia released the geforce 256 graphics card. its processor had 17 million transistors and was fabbed by tsmc. most notably, when nvidia unveiled the geforce, they called it "the world's first graphics processing unit", or gpu. a new term that must have struck people as a bit of clever marketing. at its introduction, nvidia defined the term gpu to mean: > "a single-chip processor with integrated transform, lighting, triangle setup/clipping, and rendering engines that is capable of processing a minimum of 10 million polygons per second. """,1))
# print(len(get_summary(True,"""Here i run, how are you? Thanks for coming all the way from nottingham. This sundar pichai, the ceo of google. My son is going to be excited about doing this. You watch the videos. Your review is what he cares about. Oh my god. And he's about to show us what the future of small fans looks like. He's the one who will introduce your videos to me. He's like, i love mr. who's ballad. We're all curious. We've seen phones drastically changing every single year. So suddenly, it seemed like they stopped. Congrats, i heard you got engaged this year. Yes, yes. We're planning on wedding now. That's awesome. Except i've traveled halfway across the world to find out from the one person who actually does have the answers. This is going to be fun. Have you watched any of them? Of course, yeah. Which is your favorite? You picked your seven, broad? Ha, it makes me so happy. Not. Good. Ok, so what we're really interested in is the future of the small fan. Google's launched a foldable fan. Why? It's a good question. We are trying to push the frontier of what's possible. At the end of the day, these are computing devices. You're trying to do a lot with them. You want to be very productive. You want them to entertain you. And giving people more functionality in a device they are carrying is what we are trying to do. Something confuses me about foldable. So i've got one in my pocket here. If you look at this long factor, you've got glass. You've got components. You've got glass. You've got components. Glass components. Is this the finished product? Yeah, you've got two sets of cameras. Two displays. You can only have a use one at once. Is this it? It's a good question. Look, i think it will be it for some people. The benefits it gives will outweigh some of the plate-offs they have. They are bigger and bulkier. Though this year we made it thin enough that the front screen almost is like my regular phone. I don't feel that trade-off when i use it. But i love and i can multitask on two apps. I love when i can put it as a table top. So you're actually using a fold. I've been testing a fold for a while now. I use both. There are things i'm like, well, i'd rather have my regular phone. There are times i'm like, oh, i love what the fold does. So in what kind of situations would you rather have your regular phone? I'm just traveling and i'm busy in a day. And all i'm doing is pulling out and quick checking email. I'm like, well, i would rather take a lighter phone. But you also see this as like a transitory thing. So this isn't the destination. This is the journey towards something in the future. That's right. This is for people who want to live on that future back, right? It gives you these amazing capabilities. But i do think there's more to be done here in this category. So you're the rules and said why? I guess for you, what is the future of the smartphone? Because ai was talked a lot about here. Is that the future as far as you see it? More on the software and as opposed to the hardware innovation? I think ai will make it so much more natural and intuitive to interact, which isn't today. We are the early stages of all of this. But just with natural language or when you look at something and you want your phone to understand it, that's the direction of a computing will go. We've always had humans adapt to computing versus the other way about and ai is what will actually enable computers to actually adapt to humans. The way we look and the way we talk are adapting to an interface in a computer. You were talking to how people have to figure out how to search for things. That's right. We've learned to type in a really unnatural way to get the answers we want. Whereas it's moving towards just being able to type the sentences the way you would say them. You know, we see this in some emerging markets if people haven't used phones before, they do a lot of their queries by voice because they don't have this preconceived notions of how to do it. So it's an exciting direction. But the end form factor, the future version of these phones or a pair of glasses, all that is to be played up. How does that ai glasses and stuff relate to foldable? Do you think that's a direct replacement? I don't think so. I think you will have a primary computing device and an increasingly phone server for people. What we are doing with the fold is pushing the boundaries of what a phone can be, giving them expanded capabilities in a phone like form factor. You know, there'll be other things to go with the phone. Watch us are one example. Glasses will be one down the line. And see imagine glasses as being like an accessory to the phone, at least preliminary. That's right. That is more near term than like a fully immersive computing device on your head all the time, just because of the state of technology is today. It's really hard to imagine right now that being a thing. Yeah, that's right. You think it's a way of stuff. It depends on a lot of planes today. It's good to do the sound doors, i guess. But you know, it depends on a lot. If you're really into gaming, may not be too far away where because of the immersion it offers, it kind of meets that product market effect. But for a general purpose computing device, i think we still have a while to go. So send us that foldables aren't really the end destination, but that small phones are here to stay as people's main devices. Which i agree with, we've developed such deep habits with our small phones that it's going to be really hard for any new piece of tech to replace them. But then what is the future of votes? Ok, so ai, right. Ai is a huge part of its picture now. I was watching google announce the magic editor, and i literally got goosebumps watching. I thought it was incredible, and i'm so excited about it. But at the same time, it does make me think, does any part of you worry about the inauthenticity of photos at that point? Which is why i think it's important in the context of google photos. This is designed for, hey, i didn't quite get destroyed. Maybe there was an awkward bag in the middle. I looked me brush it all up. It's to create memories. It's a good question. Where is the line? Yeah. Which is why when we talk about public images at large, they're also talking about watermarking metadata, making sure the world understands something is ai-generated. Today, you're always in your personal life. If someone is coming, you're like, quick, adjusting a living room to make it nice. Right. Is that authentic or not? Right? You care about those things. And so i think you're just giving people the power to do that in the context of their digital memories. But you're right. They think there is a continuum in these things. I trust people to figure this out. There's a lot of talk about kind of doing this responsibly. And i think it's totally right to take it slow and not rush to the end of the way to do that. One of the things that i worry about is not even the fact that this ai can fall into the hands of bad actors. But it's more just the fact that even in a best case situation, if ai does really well and it does exactly what we're trying to make it do, then what happens to us is humans. Every kid born into the world accepts every bit of technology that exists at the time when they were born. So a kid being born into the world of ai where they can have their essays written for them, have their emails written for them. What do they do in their life? Like, what are they? We will always worry as humanity about every new technology currently. Calculators have the kids who are sent mad or better or whatever it is. Humans are incredibly resourceful, creative, resilient, and they adapt. Ai is definitely powerful technology. I once said ai is the most profound technology humanity is working on. More profound than fire or electricity or anything that we have done in the past. So i understand the sentiment behind the question. I wonder about the same question all the time. But i think done correctly can liberate you to channel whatever you want to do and your powerful toolstates enable you to do that. How we find meaning, i mean, these are deep questions. But i think we will value and cherish those human experiences. If you're a doctor, you're spending a lot of time doing everything other than actually spending time with the patient and talking as it free you up so that your time is more spent on those moments. It's a very hard line, isn't it? But ai is developing very quickly. It almost feels like we're chasing after it, like trying to keep on top of it. I guess i worry that a school curriculum, for example, can very easily factor in the calculators' existence by just making calculators papers. How does a school factor in that people can self-write their essays? How do tools keep up with that? Well, you know, you could imagine maybe a teacher is judging people by getting them together in a class and asking them to discuss a topic like, those are all adaptations that are possible. These questions were asked about google's search, the fact you can find anything online. Like, what does it mean to have that information? I watch youtube all the time to learn on any topic that exists in the world. I would think that's a good thing. You're now making this accessible to pretty much everyone on the world. And i think that's good, but you point about the pace of change being very fast. I think that's real, and you have to give time for society to adapt to it. And that's going to be the tension as we make progress. I'm not sure. Senna is definitely a tech optimist. But i do agree with this point. Even though there are a million ways that ai could cause societal problems, our best chance of avoiding that is to take our time developing it. I've got a few rapid-fire questions to finish off. What for in the years? Now it's the pixel 7 pro. Okay, but i'm testing. I use everything from a samsung galaxy to the new pixel 4 to the iphone. And your sim goes all over the place. I just have extra numbers too. Okay, light mode or dark mode? Depends. I love dark mode, but then i occasionally miss being in light mode. I switch back, and i go back and forth. I'm still on the fence. Okay. But dark mode on the average. If you have to pick between a bigger battery and a better camera. Better camera. Why would a wireless charging? Wireless charging. Okay, you find that more convenient. Yeah. Okay, it's kind of a joke, semi-serious. There's currently a petition right now to make me one of the voices of google assistant. Oh, i'm a hoot. Where can i hand my resume? I was so hard to say you should hand it to bard, but you can send it to me later. So we can start with, okay. That sounds good. You have a good voice. It may work out.""",2))  )
