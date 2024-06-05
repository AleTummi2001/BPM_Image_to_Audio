import numpy as np
import cv2
import requests
from io import BytesIO
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import models, transforms
from sklearn.cluster import KMeans
import webcolors
from transformers import VitsModel, AutoTokenizer
import IPython.display as ipd
import soundfile as sf
import os

def preparazioneImmagine(img_path):
    response = requests.get(img_path)
    if response.status_code != 200:
        raise ValueError(f"Impossibile aprire l'immagine")
    img = Image.open(BytesIO(response.content))
    img = img.resize((250, 150)) #dimensione univoca
    return img

def splitImage(img): #separa sfondo da primo piano
    img = np.array(img) #trasformo l'immagine in un array di interi in modo da processarla meglio
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img.shape[:2], np.uint8)
    # Impostazione iniziale della maschera
    rect = (10, 10, img.shape[1] - 20, img.shape[0] - 20)
    mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 3
    # Definisco bordi come sicuro background
    mask[0:10, :] = 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(img).unsqueeze(0)
    modello = models.segmentation.deeplabv3_resnet101(pretrained=True)
    modello.eval()
    with torch.no_grad():
        output = modello(img_tensor)['out'][0]
        mask = output.argmax(0)
    # Converti la maschera in un array numpy
    mask = mask.byte().cpu().numpy()
    # Applica la maschera al background
    bg = img.copy()
    bg[mask == 0] = 0
    # Calcola il foreground sottraendo il background
    fg = img - bg
    #riconverto le immagini da b&w a colori
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
    fg = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB)
    return bg, fg

def ShowFgBg(fg,bg,image):
    plt.imshow(image)
    plt.show()
    plt.imshow(fg)
    plt.show()
    plt.imshow(bg)
    plt.show()

def removedBlur(bg): #rimuove la sfocatura per rendere facilitare la lettura dell'immagine da parte dei modelli
    # Applico un filtro di deconvoluzione per rimuovere la sfocatura dello sfondo e renderlo più leggibile
    # in fase di lettura
    bg = cv2.filter2D(bg, -1, kernel=np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]))
    return bg

def blackToTrasparent(bg):
    pixel_neri = (bg[:, :, 0] == 0) & (bg[:, :, 1] == 0) & (bg[:, :, 2] == 0)
    bg[pixel_neri] = [255, 255, 255]
    return bg

def fillBackground(bg, fg):
    # Legge l'immagine della maschera in primo piano in scala di grigi
    mask = cv2.imread(fg, cv2.IMREAD_GRAYSCALE)
    # Assicura che la maschera abbia le stesse dimensioni dello sfondo
    if bg.shape[:2] != mask.shape:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # riempio lo sfondo utilizzando la maschera
    output = cv2.inpaint(bg, mask, inpaintRadius=70, flags=cv2.INPAINT_TELEA)
    return output

# Trova il nome del colore CSS3 più vicino a un dato colore RGB
def closestColor(requested_color):
    min_colors = {}
    # Itera attraverso tutti i nomi dei colori CSS3 e i loro valori esadecimali corrispondenti
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        # Converte il valore esadecimale in RGB
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        # Calcola la distanza quadrata tra il colore richiesto e il colore corrente
        rd = (r_c - requested_color[0]) ** 2
        gd = (g_c - requested_color[1]) ** 2
        bd = (b_c - requested_color[2]) ** 2
        min_colors[(rd + gd + bd)] = name
    return min_colors[min(min_colors.keys())]


def getColorName(rgb_color): #converte il valore rgb nel rispettivo colore
    try:
        # Prova a ottenere il nome del colore RGB
        color_name = webcolors.rgb_to_name(rgb_color)
    except ValueError:
        # Se fallisce, trova il colore più vicino
        color_name = closestColor(rgb_color)
    return color_name

# Ottiene i colori predominanti in un'immagine
def getColors(image, num_colors):
    height, width, channels = image.shape
    # Converte l'immagine in un array 2D per l'algoritmo di clustering
    image_2d = image.reshape(height * width, channels)
    # Esegue il clustering K-means per trovare i colori predominanti
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(image_2d)
    colors = kmeans.cluster_centers_
    # Ottiene i nomi dei colori trovati
    named_colors = [getColorName(color) for color in colors]
    return named_colors


def printColor(bg, img_name): # Stampa i colori predominanti in un'immagine
    string = f"The {img_name} colors predominant are: "
    colors = getColors(bg, num_colors=3)
    for color in colors:
        string += f"{color}, "
    return string


def Imagedescription(model, processor, img, img_name):
    text = f"In the {img_name} there"  # contestualizzo
    inputs = processor(img, text, return_tensors="pt")
    out1 = model.generate(**inputs, max_length=30)
    return processor.decode(out1[0], skip_special_tokens=True)


def image_to_Text(url, model, processor):
    image = preparazioneImmagine(url)
    fg, bg = splitImage(image)
    cv2.imwrite('mask.png', fg)
    bg = fillBackground(bg, 'mask.png')
    bg = removedBlur(bg)
    fg = blackToTrasparent(fg)
    ShowFgBg(fg, bg, image)
    Background = "Background"
    Foreground = "Foreground"
    BackgroundDescription = Imagedescription(model, processor, bg, Background)
    ForegroundDescription = Imagedescription(model, processor, fg, Foreground)
    ColorDominantBg = printColor(bg, Background)
    ColorDominantFg = printColor(fg, Foreground)
    finalDescription = BackgroundDescription + ". " + ForegroundDescription + ". " + ColorDominantBg + ". " + ColorDominantFg + "."
    return finalDescription

# Converte il testo in audio utilizzando un modello specifico
def Text_to_Audio(model_name, final_description):
    model = VitsModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(final_description, return_tensors="pt")
    with torch.no_grad():
        output = model(**inputs).waveform
    waveform = output.squeeze().cpu().numpy()
    ipd.Audio(waveform, rate=22050)
    sf.write('AudioOutput.wav', waveform, 22050)
    os.system('xdg-open AudioOutput.wav')
