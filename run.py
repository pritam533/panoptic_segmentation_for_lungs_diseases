from flask import Flask
from routes.main import main
import os
app = Flask(__name__, 
            static_folder='app/static',  
            template_folder='frontend')


app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True)
