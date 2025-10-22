import React, { createContext, useContext, useState, useEffect } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';

// 🧭 Define your color palette consistent with tailwind.config.js
const lightTheme = {
  mode: 'light',
  primary: '#2C2C2C',
  secondary: '#F8F8F8',
  accent: '#FF6B00',
  highlight: '#FFAB7B',
  neutral: '#A8A8A8',
  bg: '#F8F8F8',
  surface: '#FFFFFF',
  hover: '#EBEBEB',
};

const darkTheme = {
  mode: 'dark',
  primary: '#F8F8F8',
  secondary: '#2C2C2C',
  accent: '#FF6B00',
  highlight: '#FFAB7B',
  neutral: '#A8A8A8',
  bg: '#1A1A1A',
  surface: '#333333',
  hover: '#444444',
};

interface ThemeContextProps {
  isDark: boolean;
  colors: typeof lightTheme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextProps>({
  isDark: false,
  colors: lightTheme,
  toggleTheme: () => {},
});

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDark, setIsDark] = useState(false);
  const [colors, setColors] = useState(lightTheme);

  // 🧠 Load saved theme
  useEffect(() => {
    (async () => {
      try {
        const storedTheme = await AsyncStorage.getItem('theme');
        if (storedTheme === 'dark') {
          setIsDark(true);
          setColors(darkTheme);
        } else {
          setIsDark(false);
          setColors(lightTheme);
        }
      } catch (err) {
        console.warn('Theme load error:', err);
      }
    })();
  }, []);

  // 🌓 Toggle theme
  const toggleTheme = async () => {
    const newTheme = !isDark;
    setIsDark(newTheme);
    setColors(newTheme ? darkTheme : lightTheme);

    try {
      await AsyncStorage.setItem('theme', newTheme ? 'dark' : 'light');
    } catch (err) {
      console.warn('Theme save error:', err);
    }
  };

  return (
    <ThemeContext.Provider value={{ isDark, colors, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);