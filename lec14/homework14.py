import gtts
import speech_recognition as sr
import librosa
import soundfile as sf

def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech, then write it to filename.
    
    @params:
    text (str) - the text you want to synthesize
    lang (str) - the language in which you want to synthesize it
    filename (str) - the filename in which it should be saved
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)

def make_a_corpus(texts, languages, filenames):
    '''
    Create many speech files, and check their content using SpeechRecognition.
    The output files should be created as MP3, then converted to WAV, then recognized.

    @param:
    texts - a list of the texts you want to synthesize
    languages - a list of their languages
    filenames - a list of their root filenames, without the ".mp3" ending

    @return:
    recognized_texts - list of the strings that were recognized from each file
    '''
    recognized_texts = []
    r = sr.Recognizer()
    
    for text, lang, name in zip(texts, languages, filenames):
        mp3_file = name + ".mp3"
        wav_file = name + ".wav"
        
        # 1. Synthesize to MP3
        synthesize(text, lang, mp3_file)
        
        # 2. Convert MP3 to WAV
        audio, sr_rate = librosa.load(mp3_file, sr=None)
        sf.write(wav_file, audio, sr_rate)
        
        # 3. Recognize
        with sr.AudioFile(wav_file) as source:
            audio_data = r.record(source)
        try:
            result = r.recognize_google(audio_data, language=lang)
        except:
            result = ""
        
        recognized_texts.append(result)
    
    return recognized_texts