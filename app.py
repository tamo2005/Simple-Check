from flask import Flask
from src.api.routes import routes  # Import the blueprint

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(routes)

if __name__ == "__main__":
    app.run(debug=True)

