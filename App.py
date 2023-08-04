from flask import Flask
from database import db

def create_app():
    App = Flask(__name__, static_url_path='/static')
    App.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Fire_Alerts.db'
    App.config['SQLALCHEMY_BINDS'] = {
        'fire_location': 'sqlite:///Fire_Location.db'
    }
    db.init_app(App)
    # Import the 'view' blueprint
    from View import View
    App.register_blueprint(View, url_prefix="/")

    with App.app_context():
        db.create_all()
        print(f"Fire Alerts Database path: {App.config['SQLALCHEMY_DATABASE_URI']}")
        print(f"Fire Location Database path: {App.config['SQLALCHEMY_BINDS']['fire_location']}")

    return App

if __name__ == '__main__':
    App = create_app()
    App.run(debug=True, port=5000)
