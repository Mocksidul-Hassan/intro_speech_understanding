import speech_recognition as sr

def transcribe_wavefile(filename, language):
    '''
    Use sr.Recognizer.AudioFile(filename) as the source,
    recognize from that source,
    and return the recognized text.
    
    @params:
    filename (str) - the filename from which to read the audio
    language (str) - the language of the audio
    
    @returns:
    text (str) - the recognized speech
    '''
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    
    text = r.recognize_google(audio, language=language)
    return text