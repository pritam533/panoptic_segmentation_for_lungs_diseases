# from flask import Flask
# from routes.main import main
# import os
# app = Flask(__name__, 
#             static_folder='app/static',  
#             template_folder='frontend')


# app.register_blueprint(main)

# if __name__ == '__main__':
#     app.run(debug=True)


# import os

# port = int(os.environ.get("PORT", 8080))

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=port)



# import os
# from flask import Flask
# from routes.main import main

# app = Flask(__name__, 
#             static_folder='app/static',  
#             template_folder='frontend')
# app.register_blueprint(main)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     app.run(host="0.0.0.0", port=port)




import os
from flask import Flask
from routes.main import main

app = Flask(__name__)
app.register_blueprint(main)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=10000)