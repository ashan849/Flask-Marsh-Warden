// OAuth Configuration
const CLIENT_ID = import.meta.env.VITE_GOOGLE_CLIENT_ID;
const REDIRECT_URI = import.meta.env.VITE_REDIRECT_URI || 'http://localhost:5173/auth/callback';
const AUTH_SERVER_URL = import.meta.env.VITE_AUTH_SERVER_URL || '';

// Generate Google OAuth URL
export const getGoogleAuthUrl = () => {
  const params = new URLSearchParams({
    client_id: CLIENT_ID,
    redirect_uri: REDIRECT_URI,
    response_type: 'code',
    scope: 'https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile',
    access_type: 'offline',
    prompt: 'consent'
  });

  return `https://accounts.google.com/o/oauth2/v2/auth?${params.toString()}`;
};

// Exchange code for tokens (via backend to keep client_secret secure)
export const exchangeCodeForToken = async (code) => {
  try {
    const response = await fetch(`${AUTH_SERVER_URL}/api/auth/exchange-token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ code })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Token exchange failed');
    }

    return await response.json();
  } catch (error) {
    console.error('Error exchanging code:', error);
    throw error;
  }
};

// Get user info from Google (via backend)
export const getUserInfo = async (accessToken) => {
  try {
    const response = await fetch(`${AUTH_SERVER_URL}/api/auth/userinfo`, {
      headers: { Authorization: `Bearer ${accessToken}` }
    });

    if (!response.ok) {
      throw new Error('Failed to get user info');
    }

    return await response.json();
  } catch (error) {
    console.error('Error getting user info:', error);
    throw error;
  }
};

// Save user to localStorage
export const saveUser = (user) => {
  localStorage.setItem('chatbot_user', JSON.stringify({
    ...user,
    loginTime: Date.now()
  }));
};

// Load user from localStorage
export const loadUser = () => {
  const stored = localStorage.getItem('chatbot_user');
  if (!stored) return null;

  const user = JSON.parse(stored);

  // Check if login is less than 2 hours old
  const twoHours = 2 * 60 * 60 * 1000;
  if (Date.now() - user.loginTime > twoHours) {
    logoutUser();
    return null;
  }

  return user;
};

// Logout user
export const logoutUser = () => {
  localStorage.removeItem('chatbot_user');
};