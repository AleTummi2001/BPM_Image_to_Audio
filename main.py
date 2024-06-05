import function
import os
from transformers import BlipProcessor, BlipForConditionalGeneration

os.environ["TOKENIZERS_PARALLELISM"] = "false"

url1 = "https://www.marcotogni.it/foto-altri-v1/foto-sfondo-sfocato3.jpg"
url = "https://www.fotografiamoderna.it/wp-content/uploads/2019/11/come-fare-delle-foto-nitide.jpg"
url2 = "https://www.popupmag.it/wp-content/uploads/2020/05/foto-gratis-e1588931860135.jpg"
url3 = "https://www.tiscali.it/export/sites/tecnologia/.galleries/16/12-siti-per-vendere-foto.jpg"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
finalDescription1 = function.image_to_Text(url2, model, processor)
#finalDescription2 = function.image_to_Text(url2, model, processor)

print(finalDescription1)
#print(finalDescription2)
function.Text_to_Audio("facebook/mms-tts-eng", finalDescription1)



