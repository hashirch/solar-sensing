import {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  type ReactNode,
} from "react";

export const API_URL =
  (import.meta as any).env?.VITE_API_URL || "http://localhost:8000";

export interface UserProfile {
  id: number;
  email: string;
  displayName: string;
  role: string;
  createdAt: number;
  lastLoginAt: number | null;
}

type Result = { ok: boolean; error?: string; data?: any };

interface AuthContextValue {
  authenticated: boolean;
  checking: boolean;
  email: string;
  user: UserProfile | null;
  register: (name: string, email: string, password: string) => Promise<Result>;
  login: (email: string, password: string) => Promise<Result>;
  resendCode: () => Promise<Result>;
  verifyOtp: (code: string) => Promise<Result>;
  updateName: (name: string) => Promise<Result>;
  requestPasswordChange: (current: string, next: string) => Promise<Result>;
  confirmPasswordChange: (code: string) => Promise<Result>;
  authFetch: (path: string, init?: RequestInit) => Promise<Result>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);
const TOKEN_KEY = "solarsensing_token";

async function api(path: string, init: RequestInit = {}, token = ""): Promise<Result> {
  try {
    const res = await fetch(`${API_URL}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
        ...(init.headers || {}),
      },
    });
    const data = await res.json().catch(() => ({}));
    if (res.ok && data.success) return { ok: true, data };
    return { ok: false, error: data.error || "Request failed. Please try again." };
  } catch {
    return {
      ok: false,
      error: "Cannot reach the server. Make sure it is running on port 8000.",
    };
  }
}

const post = (path: string, body: object, token = "") =>
  api(path, { method: "POST", body: JSON.stringify(body) }, token);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [email, setEmail] = useState("");
  const [user, setUser] = useState<UserProfile | null>(null);
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    ["solarsensing_auth", "solarsensing_email", "solarsensing_otp"].forEach((k) =>
      localStorage.removeItem(k)
    ); // legacy, unverified values
    const token = sessionStorage.getItem(TOKEN_KEY);
    if (!token) {
      setChecking(false);
      return;
    }
    post("/api/session", { token }).then((r) => {
      if (r.ok) setUser(r.data.user);
      else sessionStorage.removeItem(TOKEN_KEY);
      setChecking(false);
    });
  }, []);

  const token = () => sessionStorage.getItem(TOKEN_KEY) || "";

  const register = useCallback(async (name: string, addr: string, password: string) => {
    const clean = addr.trim().toLowerCase();
    const r = await post("/api/auth/register", { name, email: clean, password });
    if (r.ok) setEmail(clean);
    return r;
  }, []);

  const login = useCallback(async (addr: string, password: string) => {
    const clean = addr.trim().toLowerCase();
    const r = await post("/api/auth/login", { email: clean, password });
    if (r.ok && r.data.token) {
      sessionStorage.setItem(TOKEN_KEY, r.data.token);
      setEmail(clean);
      setUser(r.data.user);
    }
    return r;
  }, []);

  const resendCode = useCallback(() => post("/api/otp/request", { email }), [email]);

  const verifyOtp = useCallback(
    async (code: string) => {
      const r = await post("/api/otp/verify", { email, code });
      if (r.ok && r.data.token) {
        sessionStorage.setItem(TOKEN_KEY, r.data.token);
        setUser(r.data.user);
        return r;
      }
      return r.ok ? { ok: false, error: "Verification failed." } : r;
    },
    [email]
  );

  const updateName = useCallback(async (name: string) => {
    const r = await post("/api/me", { displayName: name }, token());
    if (r.ok) setUser(r.data.user);
    return r;
  }, []);

  const requestPasswordChange = useCallback(
    (currentPassword: string, newPassword: string) =>
      post("/api/me/password/request", { currentPassword, newPassword }, token()),
    []
  );

  const confirmPasswordChange = useCallback(
    (code: string) => post("/api/me/password/confirm", { code }, token()),
    []
  );

  const authFetch = useCallback(
    (path: string, init: RequestInit = {}) => api(path, init, token()),
    []
  );

  const logout = useCallback(() => {
    fetch(`${API_URL}/api/logout`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token()}` },
    }).catch(() => undefined);
    sessionStorage.removeItem(TOKEN_KEY);
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider
      value={{
        authenticated: !!user,
        checking,
        email,
        user,
        register,
        login,
        resendCode,
        verifyOtp,
        updateName,
        requestPasswordChange,
        confirmPasswordChange,
        authFetch,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
