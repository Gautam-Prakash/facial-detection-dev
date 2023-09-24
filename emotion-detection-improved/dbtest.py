from flask import Flask, request, flash, redirect, url_for, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://app_user:app_password@localhost/app_database'  # replace 'password' with the actual password
db = SQLAlchemy(app)

app.secret_key = 'your_secret_key_here'

class User(db.Model):
    __tablename__ = 'users'  # Specify the table name since it's not the default plural form

    user_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    phone_number = db.Column(db.String, unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())

class Result(db.Model):
    __tablename__ = 'results'  # Specify the table name since it's not the default plural form

    result_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    emotion_data = db.Column(db.JSON)

# Add a new user
@app.route('/add_user/<phone>')
def add_user(phone):
    new_user = User(phone_number=phone)
    db.session.add(new_user)
    db.session.commit()
    return f"User {phone} added successfully!"

# Query all users
@app.route('/list_users')
def list_users():
    users = User.query.all()
    return '\n'.join([user.phone_number for user in users])

@app.route('/test')
def test():
    return "Test successful!"

@app.route('/login_page')
def login_page():
    return render_template('login.html')


@app.route('/login', methods=['POST'])
def login():
    phone_number = request.form.get('phone_number')
    
    # Check if the phone number already exists
    existing_user = User.query.filter_by(phone_number=phone_number).first()
    if existing_user:
        flash('Phone number already exists!')
        return redirect(url_for('login_page'))
    
    # Insert the new phone number into the users table
    new_user = User(phone_number=phone_number)
    db.session.add(new_user)
    db.session.commit()
    
    flash('User added successfully!')
    return redirect(url_for('login_page'))

if __name__ == '__main__':
        app.run(host='127.0.0.1', port=5001)

# from flask import Flask

# app = Flask(__name__)

# @app.route('/test')
# def test():
#     return "Test successful!"

# if __name__ == '__main__':
#     app.run(host='127.0.0.1', port=5001)
