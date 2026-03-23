from flask import Flask
from routes.main import main
import os
app = Flask(__name__, 
            static_folder='app/static',  
            template_folder='app/templates')
# app = Flask(__name__, template_folder=os.path.abspath('app/templates'))

# ... rest of your code ...
app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True)

import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)