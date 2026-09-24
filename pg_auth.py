"""
PulseGuard authentication and user management.

* Users live in Supabase (table app_users); passwords are bcrypt-hashed.
* Admin and Analyst accounts must pass authenticator-app (TOTP) MFA.
* Admins manage users from the dashboard; every action is written to audit_log.

The database is reached over Supabase's REST API with the server-side secret key
from st.secrets["supabase"]. The key never reaches the browser.
"""
import io
from datetime import datetime, timedelta, timezone

import bcrypt
import pandas as pd
import pyotp
import qrcode
import requests
import streamlit as st

ROLES = ["Admin", "Analyst", "Read-only"]
DEFAULT_MFA_ROLES = ["Admin", "Analyst"]   # override with [auth] mfa_roles = [...] in secrets
MIN_PASSWORD = 10
MAX_FAILS = 5          # failed attempts allowed ...
LOCK_MINUTES = 15      # ... within this many minutes, per username
SESSION_RECHECK_SECONDS = 60
ISSUER = "PulseGuard"


# ── Database layer ────────────────────────────────────────────────────────────
class DBError(Exception):
    pass


def _cfg():
    try:
        s = st.secrets["supabase"]
        return str(s["url"]).rstrip("/"), str(s["key"]).strip()
    except Exception:
        return None, None


def _req(method, table, params=None, body=None, prefer=None):
    url, key = _cfg()
    if not url or not key:
        raise DBError("Supabase secrets are missing ([supabase] url and key).")
    headers = {"apikey": key, "Content-Type": "application/json"}
    if prefer:
        headers["Prefer"] = prefer
    endpoint = f"{url}/rest/v1/{table}"
    try:
        r = requests.request(method, endpoint, headers=headers, params=params, json=body, timeout=15)
        if r.status_code in (401, 403):
            # Some key formats also need the Authorization header; retry once.
            headers["Authorization"] = f"Bearer {key}"
            r = requests.request(method, endpoint, headers=headers, params=params, json=body, timeout=15)
    except requests.RequestException as e:
        raise DBError(
            f"Could not reach the user database ({type(e).__name__}). "
            "If the Supabase project has been idle for a week it may be paused: "
            "unpause it from the Supabase dashboard."
        )
    if r.status_code >= 400:
        raise DBError(f"Database error {r.status_code}: {r.text[:200]}")
    return r.json() if r.text else []


def _eq(value):
    return f"eq.{value}"


def get_user(username):
    if not username:
        return None
    rows = _req("GET", "app_users", params={"username": _eq(username), "select": "*", "limit": "1"})
    return rows[0] if rows else None


def count_users():
    return len(_req("GET", "app_users", params={"select": "username", "limit": "1"}))


def list_users():
    return _req("GET", "app_users", params={
        "select": "username,role,mfa_enabled,created_at,created_by", "order": "created_at.asc"})


def create_user(username, password, role, created_by):
    _req("POST", "app_users", prefer="return=minimal", body={
        "username": username, "password_hash": hash_pw(password),
        "role": role, "created_by": created_by})


def update_user(username, fields):
    _req("PATCH", "app_users", params={"username": _eq(username)}, body=fields, prefer="return=minimal")


def delete_user(username):
    _req("DELETE", "app_users", params={"username": _eq(username)}, prefer="return=minimal")


def log(actor, action, target=None, detail=None):
    """Audit trail. Never raises: logging must not break the app."""
    try:
        _req("POST", "audit_log", prefer="return=minimal", body={
            "actor": (actor or "?")[:80], "action": action,
            "target": (target[:80] if target else None), "detail": detail})
    except DBError:
        pass


def _recent_fails(username):
    since = (datetime.now(timezone.utc) - timedelta(minutes=LOCK_MINUTES)).isoformat()
    rows = _req("GET", "audit_log", params={
        "action": "eq.login_failed", "target": _eq(username[:80]),
        "at": f"gte.{since}", "select": "id", "limit": str(MAX_FAILS)})
    return len(rows)


# ── Passwords and MFA helpers ────────────────────────────────────────────────
def hash_pw(password):
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def check_pw(password, hashed):
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], hashed.encode("utf-8"))
    except Exception:
        return False


_DUMMY = None


def _dummy_hash():
    """Compare against a dummy hash for unknown usernames so timing doesn't reveal them."""
    global _DUMMY
    if _DUMMY is None:
        _DUMMY = hash_pw("not-a-real-password")
    return _DUMMY


def _pw_problem(password):
    if len(password or "") < MIN_PASSWORD:
        return f"Password must be at least {MIN_PASSWORD} characters."
    return None


def _mfa_roles():
    try:
        return list(st.secrets["auth"]["mfa_roles"])
    except Exception:
        return list(DEFAULT_MFA_ROLES)


def seed_from_secrets():
    """First run only: copy the accounts from the old [users] secrets into the database."""
    try:
        old = dict(st.secrets["users"])
    except Exception:
        return 0
    added = 0
    for name, info in old.items():
        info = dict(info)
        if info.get("password") and info.get("role") in ROLES:
            try:
                create_user(name, info["password"], info["role"], "seed")
                added += 1
            except DBError:
                pass
    if added:
        log("system", "seed_users", detail=f"{added} accounts copied from secrets")
    return added


def _ensure_seeded():
    if st.session_state.get("pg_ready"):
        return None
    if count_users() == 0 and seed_from_secrets() == 0:
        return "No accounts exist yet, and there is no [users] section in secrets to copy from."
    st.session_state["pg_ready"] = True
    return None


# ── Login flow ────────────────────────────────────────────────────────────────
def _finish_login(user):
    st.session_state.authenticated = True
    st.session_state.role = user["role"]
    st.session_state.username = user["username"]
    st.session_state["pg_checked"] = datetime.now(timezone.utc).timestamp()
    for k in ("pg_stage", "pg_pending", "pg_enroll_secret"):
        st.session_state.pop(k, None)
    log(user["username"], "login_ok")
    st.rerun()


def _cancel_mfa():
    for k in ("pg_stage", "pg_pending", "pg_enroll_secret"):
        st.session_state.pop(k, None)
    st.rerun()


def render_login_form():
    """Drop-in replacement for the old username/password form."""
    try:
        problem = _ensure_seeded()
    except DBError as e:
        st.error(str(e))
        return
    if problem:
        st.error(problem)
        return

    stage = st.session_state.get("pg_stage")
    if stage in ("verify", "enroll"):
        _render_mfa_step(stage)
        return

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log In")
    if submitted:
        _handle_password(username.strip(), password)


def _handle_password(username, password):
    try:
        if username and _recent_fails(username) >= MAX_FAILS:
            st.error(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.")
            return
        user = get_user(username) if username else None
        ok = check_pw(password, user["password_hash"] if user else _dummy_hash())
        if not (user and ok):
            log(username or "?", "login_failed", target=username or "?", detail="password")
            st.error("Incorrect username or password")
            return
    except DBError as e:
        st.error(str(e))
        return

    if user["role"] in _mfa_roles():
        st.session_state.pg_pending = user["username"]
        st.session_state.pg_stage = "verify" if (user.get("mfa_enabled") and user.get("mfa_secret")) else "enroll"
        st.rerun()
    _finish_login(user)


def _render_mfa_step(stage):
    uname = st.session_state.get("pg_pending")
    try:
        user = get_user(uname)
    except DBError as e:
        st.error(str(e))
        return
    if not user:
        _cancel_mfa()
        return

    if stage == "enroll":
        secret = st.session_state.setdefault("pg_enroll_secret", pyotp.random_base32())
        uri = pyotp.TOTP(secret).provisioning_uri(name=uname, issuer_name=ISSUER)
        st.markdown("**Set up two-factor authentication**")
        st.caption("Scan this QR code with an authenticator app (Google Authenticator, "
                   "Microsoft Authenticator, Authy or 1Password), then enter the 6-digit code it shows.")
        buf = io.BytesIO()
        qrcode.make(uri).save(buf, format="PNG")
        st.image(buf.getvalue(), width=200)
        st.caption("Can't scan? Enter this key in the app manually:")
        st.code(secret, language=None)
        active_secret = secret
    else:
        st.markdown("**Two-factor authentication**")
        st.caption("Enter the 6-digit code from your authenticator app.")
        active_secret = user.get("mfa_secret")

    with st.form("mfa_form"):
        code = st.text_input("6-digit code", max_chars=6)
        c1, c2 = st.columns(2)
        go = c1.form_submit_button("Verify")
        back = c2.form_submit_button("Cancel")
    if back:
        _cancel_mfa()
        return
    if go:
        try:
            if _recent_fails(uname) >= MAX_FAILS:
                st.error(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.")
                return
            if active_secret and pyotp.TOTP(active_secret).verify(code.strip(), valid_window=1):
                if stage == "enroll":
                    update_user(uname, {"mfa_secret": active_secret, "mfa_enabled": True})
                    log(uname, "mfa_enrolled", target=uname)
                _finish_login(user)
            else:
                log(uname, "login_failed", target=uname, detail="mfa")
                st.error("Invalid code. Check the time on your phone and try again.")
        except DBError as e:
            st.error(str(e))


def session_guard():
    """Log a user out (within about a minute) if their account was removed or their role changed."""
    now = datetime.now(timezone.utc).timestamp()
    if now - st.session_state.get("pg_checked", 0) < SESSION_RECHECK_SECONDS:
        return
    try:
        u = get_user(st.session_state.get("username", ""))
    except DBError:
        return  # a temporary outage should not lock everyone out
    if not u or u["role"] != st.session_state.get("role"):
        st.session_state.authenticated = False
        st.session_state.role = None
        st.session_state.username = None
        st.session_state.pop("pg_checked", None)
        st.rerun()
    st.session_state["pg_checked"] = now


# ── Sidebar: change own password ─────────────────────────────────────────────
def render_change_password():
    with st.expander("Change password"):
        with st.form("chpw_form", clear_on_submit=True):
            cur = st.text_input("Current password", type="password")
            new = st.text_input("New password", type="password")
            new2 = st.text_input("Repeat new password", type="password")
            go = st.form_submit_button("Update")
        if go:
            me = st.session_state.get("username")
            try:
                user = get_user(me)
                if not user or not check_pw(cur, user["password_hash"]):
                    st.error("Current password is incorrect.")
                elif new != new2:
                    st.error("New passwords do not match.")
                elif _pw_problem(new):
                    st.error(_pw_problem(new))
                else:
                    update_user(me, {"password_hash": hash_pw(new)})
                    log(me, "password_changed", target=me)
                    st.success("Password updated.")
            except DBError as e:
                st.error(str(e))


# ── Admin page: user management ──────────────────────────────────────────────
def _flash(kind, message):
    st.session_state["pg_flash"] = (kind, message)
    st.rerun()


def _require_admin():
    """Check the database, not just the session, before any admin action."""
    me = get_user(st.session_state.get("username", ""))
    if not me or me["role"] != "Admin":
        st.error("This page is restricted to Admin accounts.")
        st.stop()
    return me


def _admin_action(action, ok_message):
    try:
        _require_admin()
        action()
    except DBError as e:
        msg = str(e)
        if "409" in msg:
            msg = "That username already exists."
        _flash("error", msg)
    _flash("success", ok_message)


def render_user_management():
    st.markdown("## User Management")
    st.markdown("<div style='color:#8B93A1;font-size:13px;'>Add or remove dashboard users, change roles, "
                "and reset passwords or authenticators. Every action is recorded in the audit log.</div>",
                unsafe_allow_html=True)
    st.markdown("---")

    flash = st.session_state.pop("pg_flash", None)
    if flash:
        (st.success if flash[0] == "success" else st.error)(flash[1])

    try:
        me = _require_admin()
        users = list_users()
    except DBError as e:
        st.error(str(e))
        st.stop()

    me_name = me["username"]
    admins = [u for u in users if u["role"] == "Admin"]
    mfa_roles = _mfa_roles()

    # -- current users
    st.markdown("#### Current users")
    rows = []
    for u in users:
        if u["mfa_enabled"]:
            mfa = "On"
        else:
            mfa = "Set up at next login" if u["role"] in mfa_roles else "Not required"
        rows.append({"Username": u["username"], "Role": u["role"], "MFA": mfa,
                     "Created": (u.get("created_at") or "")[:16].replace("T", " "),
                     "Created by": u.get("created_by") or ""})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # -- add user
    st.markdown("#### Add a user")
    with st.form("pg_add_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 1, 2])
        new_name = c1.text_input("Username")
        new_role = c2.selectbox("Role", ROLES, index=2)
        new_pw = c3.text_input("Temporary password", type="password",
                               help=f"At least {MIN_PASSWORD} characters. Share it privately; "
                                    "the user can change it after logging in.")
        add = st.form_submit_button("Add user")
    if add:
        new_name = new_name.strip()
        if not new_name:
            _flash("error", "Enter a username.")
        elif _pw_problem(new_pw):
            _flash("error", _pw_problem(new_pw))
        elif any(u["username"].lower() == new_name.lower() for u in users):
            _flash("error", "That username already exists.")
        else:
            def _add():
                create_user(new_name, new_pw, new_role, me_name)
                log(me_name, "user_added", target=new_name, detail=f"role={new_role}")
            _admin_action(_add, f"Added {new_name} as {new_role}.")

    # -- change role / reset password / reset MFA
    st.markdown("#### Change a user")
    names = [u["username"] for u in users]
    target = st.selectbox("User", names, key="pg_target")
    tu = next(u for u in users if u["username"] == target)

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.form("pg_role_form"):
            chosen_role = st.selectbox("New role", ROLES, index=ROLES.index(tu["role"]),
                                       key=f"pg_nr_{target}")
            go_role = st.form_submit_button("Apply role")
    with c2:
        with st.form("pg_pw_form", clear_on_submit=True):
            reset_pw = st.text_input("New temporary password", type="password", key=f"pg_rp_{target}")
            go_pw = st.form_submit_button("Reset password")
    with c3:
        st.caption("Clears the user's authenticator so they set up a new one at next login.")
        go_mfa = st.button("Reset MFA", key=f"pg_rm_{target}")

    if go_role:
        if target == me_name:
            _flash("error", "You can't change your own role. Ask another admin.")
        elif chosen_role == tu["role"]:
            _flash("error", f"{target} is already {chosen_role}.")
        elif tu["role"] == "Admin" and len(admins) <= 1:
            _flash("error", "That would leave no Admin accounts.")
        else:
            def _role():
                update_user(target, {"role": chosen_role})
                log(me_name, "role_changed", target=target, detail=f"{tu['role']} -> {chosen_role}")
            _admin_action(_role, f"{target} is now {chosen_role}.")
    if go_pw:
        if _pw_problem(reset_pw):
            _flash("error", _pw_problem(reset_pw))
        else:
            def _pw():
                update_user(target, {"password_hash": hash_pw(reset_pw)})
                log(me_name, "password_reset", target=target)
            _admin_action(_pw, f"Password reset for {target}.")
    if go_mfa:
        def _mfa():
            update_user(target, {"mfa_secret": None, "mfa_enabled": False})
            log(me_name, "mfa_reset", target=target)
        _admin_action(_mfa, f"Authenticator cleared for {target}.")

    # -- remove user
    st.markdown("#### Remove a user")
    others = [n for n in names if n != me_name]
    if not others:
        st.caption("There are no other users to remove.")
    else:
        with st.form("pg_del_form"):
            del_name = st.selectbox("User to remove", others)
            sure = st.checkbox("I understand this removes their access immediately.")
            go_del = st.form_submit_button("Remove user")
        if go_del:
            victim = next(u for u in users if u["username"] == del_name)
            if not sure:
                _flash("error", "Tick the box to confirm.")
            elif victim["role"] == "Admin" and len(admins) <= 1:
                _flash("error", "That would leave no Admin accounts.")
            else:
                def _del():
                    delete_user(del_name)
                    log(me_name, "user_removed", target=del_name, detail=f"role={victim['role']}")
                _admin_action(_del, f"Removed {del_name}.")

    # -- audit log
    with st.expander("Audit log (latest 50)"):
        try:
            entries = _req("GET", "audit_log", params={
                "select": "at,actor,action,target,detail", "order": "at.desc", "limit": "50"})
            if entries:
                df = pd.DataFrame(entries)
                df["at"] = df["at"].astype(str).str[:19].str.replace("T", " ")
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.caption("No entries yet.")
        except DBError as e:
            st.error(str(e))
