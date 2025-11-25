import React, { createContext, useContext, useState, useEffect } from 'react';
import { getCurrentUserId, fetchUserSettings, updateUserSetting } from '../utils/supabaseApi';
import { supabase } from './supabaseClient';

interface SettingsContextProps {
  vibrationEnabled: boolean;
  toggleVibration: () => void;
}

const SettingsContext = createContext<SettingsContextProps>({
  vibrationEnabled: true,
  toggleVibration: () => {},
});

export const SettingsProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [vibrationEnabled, setVibrationEnabled] = useState(true); // Default to true
  const [userId, setUserId] = useState<string | null>(null);

  // 1. Initial Load & Auth Listener
  useEffect(() => {
    const loadSettings = async () => {
      const uid = await getCurrentUserId();
      setUserId(uid);
      if (uid) {
        const data = await fetchUserSettings(uid);
        if (data) {
          // If data exists in DB, use it. Otherwise default is true.
          if (data.vibration_enabled !== undefined) {
            setVibrationEnabled(data.vibration_enabled);
          }
        }
      }
    };

    loadSettings();

    // Listen for auth changes (login/logout) to reload settings
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        setUserId(session.user.id);
        // Reload settings for new user
        fetchUserSettings(session.user.id).then(data => {
          if (data && data.vibration_enabled !== undefined) {
            setVibrationEnabled(data.vibration_enabled);
          }
        });
      } else {
        setUserId(null);
        setVibrationEnabled(true); // Reset to default on logout
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  // 2. Toggle Function
  const toggleVibration = async () => {
    const newValue = !vibrationEnabled;
    setVibrationEnabled(newValue); // Optimistic update

    if (userId) {
      await updateUserSetting(userId, 'vibration_enabled', newValue);
    }
  };

  return (
    <SettingsContext.Provider value={{ vibrationEnabled, toggleVibration }}>
      {children}
    </SettingsContext.Provider>
  );
};

export const useSettings = () => useContext(SettingsContext);