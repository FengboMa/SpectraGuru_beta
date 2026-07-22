import streamlit as st
import streamlit.components.v1 as components
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
clerk_component_path = os.path.join(current_dir, "frontend", "build")
LOCAL_DEPLOY = not os.path.isdir(clerk_component_path)

if LOCAL_DEPLOY:
    _clerk_component = None
else:
    _clerk_component = components.declare_component(
        "clerk_component",
        #url="http://localhost:3001/",
        path=clerk_component_path
    )

def clerk_component(key, action, height_offset=25, min_height=100, visible=True):
    if LOCAL_DEPLOY:
        return "NO_USER"
    return _clerk_component(key=key, action=action, height_offset=height_offset, min_height=min_height, visible=visible)

def populate(user):
    if user and not st.session_state.user_logged_in:
        st.session_state.user_decided = True
        if not user == "NO_USER":
            st.session_state.user = user
            st.session_state.user_logged_in = user.get('signedIn', False)
            st.session_state.username = user.get('firstName') or "Guest"

            st.session_state.show_login_modal = False
            st.session_state.show_welcome_modal = False

            return True
        else:
            return False
    return False


# checks whether the user is already logged in and populates the user dict accordingly.
def startup():
    if LOCAL_DEPLOY:
        st.session_state.user_decided = True
        st.session_state.do_startup = False
        return

    placeholder = st.empty()
    with placeholder:
        user = clerk_component(key="startup", action="startup", visible=False)

        populate(user)

    if st.session_state.user_decided:
        placeholder.empty()
        st.session_state.do_startup = False
    else:
        st.write("Loading user data...")
        st.stop() # do not go forward without getting confirmation from Clerk about user login status

def login():
    if LOCAL_DEPLOY:
        st.session_state.show_login_modal = False
        return

    if 'global_placeholder' in st.session_state:
        st.session_state.global_placeholder.empty()

    st.session_state.show_login_modal = True
    
def logout():
    if LOCAL_DEPLOY:
        st.session_state.user_decided = True
        st.session_state.user = None
        st.session_state.user_logged_in = False
        return

    with st.session_state.global_placeholder:
        clerk_component(key="logout", action="logout", visible=False)
    
    st.session_state.user_decided = False
    st.session_state.user = None
    st.session_state.user_logged_in = False

# Forces the user to be logged in to continue. If not logged in, a login popup appears.
# This function should be called at the beginning of each page to make it inaccessible to Guest users.
def force_login():
    pass

