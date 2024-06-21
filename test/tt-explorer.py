from flask import Flask, render_template
import model_explorer
import threading

app = Flask(__name__)

model_explorer_config = model_explorer.config()
model_explorer_url = 'http://localhost'

# Configure the model_explorer to load some arbritrary model file:

model_explorer_config.add_model_from_path('./1.tflite')

def run_model_explorer(model_explorer_port=5006):
    global model_explorer_url
    model_explorer_url += f':{model_explorer_port}/?show_open_in_new_tab=1&data={model_explorer_config.to_url_param_value()}'
    model_explorer.visualize_from_config(
        config=model_explorer_config,
        no_open_in_browser=True,
        port=model_explorer_port,
    )


# Create a thread to run the model explorer
model_explorer_thread = threading.Thread(target = run_model_explorer)
model_explorer_thread.start()

@app.route('/')
def index(): 
   return render_template('index.html', url=model_explorer_url)

app.run()