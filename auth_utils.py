import streamlit as st
import streamlit.components.v1 as components
from streamlit_modal import Modal

_clerk_component = components.declare_component(
    "clerk_component",
    url="http://localhost:3001/"
)

def clerk_component(key, action, height=0):
    return _clerk_component(key=key, action=action, height=height)

def populate(user):
    if user:
        st.session_state.user_decided = True
        if not user == "NO_USER":
            st.session_state.user = user
            st.session_state.user_logged_in = user['signedIn']

            st.session_state.show_login_modal = False
            st.session_state.show_welcome_modal = False

            return True
        else:
            return False


# checks whether the user is already logged in and populates the user dict accordingly.
def startup():
    placeholder = st.empty()
    with placeholder:
        user = clerk_component(key="startup", action="startup")

        populate(user)

    if st.session_state.user_decided:
        placeholder.empty()
        st.session_state.do_startup = False
    else:
        st.write("Loading user data...")
        st.stop() # do not go forward without getting confirmation from Clerk about user login status

def login():
    if 'global_placeholder' in st.session_state:
        st.session_state.global_placeholder.empty()

    st.session_state.show_login_modal = True
    
def logout():
    with st.session_state.global_placeholder:
        clerk_component(key="logout", action="logout")
    
    st.session_state.user_decided = False
    st.session_state.user = None
    st.session_state.user_logged_in = False