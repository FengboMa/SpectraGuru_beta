import streamlit as st
import streamlit.components.v1 as components
from streamlit_modal import Modal

_clerk_component = components.declare_component(
    "clerk_component",
    url="http://localhost:3001/"
)

def populate(user):
    if user:
        st.session_state.user_decided = True
        if not user == "NO_USER":
            st.session_state.user = user
            st.session_state.user_logged_in = user['signedIn']
            return True
        else:
            return False


# checks whether the user is already logged in and populates the user dict accordingly.
def startup():
    placeholder = st.empty()
    with placeholder:
        user = _clerk_component(key="startup", action="startup", height=0)

        populate(user)

    if st.session_state.user_decided:
        placeholder.empty()
    else:
        st.write("Loading user data...")
        st.stop() # do not go forward without getting confirmation from Clerk about user login status

def login():
    st.session_state.attempt_logout = False
    print("ATTEMPT LOGOUT FALSE LINE 37")
    st.session_state.show_welcome_modal = False
    st.session_state.show_login_modal = True
    
def logout():
    placeholder = st.empty()
    with placeholder:
        _clerk_component(key="logout", action="logout", height=0)
    
    restore_defaults()
    st.session_state.attempt_logout = True
    print("ATTEMPT LOGOUT TRUE LINE 48")

def restore_defaults():
    st.session_state.user_decided = False
    st.session_state.user = None
    st.session_state.user_logged_in = False
    st.session_state.show_welcome_modal = True
    st.session_state.show_login_modal = False