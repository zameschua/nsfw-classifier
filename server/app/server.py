from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():

    try:
        learn = load_learner('./models')
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/api/v1/classify', methods=['POST'])
def classifyEndpoint():
  if request.method == 'POST':
      """ Receive base 64 encoded image """
      print('Request received')
      request_data = json.loads(request.get_data())
      data = request_data['data'][22:]

      probability = classify(data)
      print(probability)
      response = {
        'nsfw': probability,
        'sfw': 1 - probability
      }
      return jsonify(response)

# Takes an image in bytes and returns the probability that it is NSFW
def classify(data):
  im = Image.open(BytesIO(base64.b64decode(data)))
  im.save('temp.png', 'PNG')

  img = open_image('temp.png')
  pred_class,pred_idx,outputs = learn.predict(img)

  return outputs[0].item()


if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
