import { saveUser } from './auth';

export default function Login({ onLogin }) {
  const loginWithGoogle = async () => {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || '';
      const response = await fetch(`${apiUrl}/api/auth/google/login`);
      const data = await response.json();
      if (data.auth_url) {
        window.location.href = data.auth_url;
      }
    } catch (error) {
      console.error('Failed to get auth URL:', error);
    }
  };

  const loginAsGuest = () => {
    const guestUser = { name: 'Guest User', email: 'guest@example.com', picture: 'https://cdn-icons-png.flaticon.com/512/149/149071.png', isGuest: true };
    saveUser(guestUser);
    onLogin(guestUser);
  };

  return (
    <div className="login-card">
      <button onClick={loginWithGoogle} className="google-btn">
        Sign in with Google
      </button>
      <button
        onClick={loginAsGuest}
        style={{
          marginTop: '12px',
          background: 'none',
          border: 'none',
          color: '#1a759f',
          cursor: 'pointer',
          fontSize: '14px',
          fontWeight: '500'
        }}
      >
        Login as Guest
      </button>
    </div>
  );
}
