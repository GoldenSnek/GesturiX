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

  useEffect(() => {
    const loadSettings = async () => {
      const uid = await getCurrentUserId();
      setUserId(uid);
      if (uid) {
        const data = await fetchUserSettings(uid);
        if (data) {
          if (data.vibration_enabled !== undefined) {
            setVibrationEnabled(data.vibration_enabled);
          }
        }
      }
    };

    loadSettings();

    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      if (session?.user) {
        setUserId(session.user.id);
        fetchUserSettings(session.user.id).then(data => {
          if (data && data.vibration_enabled !== undefined) {
            setVibrationEnabled(data.vibration_enabled);
          }
        });
      } else {
        setUserId(null);
        setVibrationEnabled(true);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  const toggleVibration = async () => {
    const newValue = !vibrationEnabled;
    setVibrationEnabled(newValue);

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