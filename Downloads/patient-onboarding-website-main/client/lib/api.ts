// API utilities for making authenticated requests

export class APIError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

export async function makeAuthenticatedRequest(
  url: string,
  options: RequestInit = {}
): Promise<any> {
  const token = localStorage.getItem("healthcarePlus_token");
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  const data = await response.json();

  if (!response.ok) {
    // If token is invalid, clear it and redirect to login
    if (response.status === 401 || response.status === 403) {
      localStorage.removeItem("healthcarePlus_token");
      localStorage.removeItem("healthcarePlus_currentUser");
      window.location.href = "/login";
    }
    
    throw new APIError(
      data.message || `HTTP ${response.status}`,
      response.status,
      data
    );
  }

  return data;
}

export async function makePublicRequest(
  url: string,
  options: RequestInit = {}
): Promise<any> {
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  const data = await response.json();

  if (!response.ok) {
    throw new APIError(
      data.message || `HTTP ${response.status}`,
      response.status,
      data
    );
  }

  return data;
}

// Check if user is authenticated
export function isAuthenticated(): boolean {
  return !!localStorage.getItem("healthcarePlus_token");
}

// Clear authentication
export function clearAuth(): void {
  localStorage.removeItem("healthcarePlus_token");
  localStorage.removeItem("healthcarePlus_currentUser");
}

// Save user authentication
export function saveAuth(token: string, user: any): void {
  localStorage.setItem("healthcarePlus_token", token);
  localStorage.setItem("healthcarePlus_currentUser", user.email);
}
