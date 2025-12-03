import { View } from 'react-native';
import React from 'react';
import { Tabs } from 'expo-router';
import CustomTabBar from '../../components/CustomTabBar';
import { useTheme } from '../../src/ThemeContext';

const _layout = () => {
  const { isDark } = useTheme();

  return (
    <View style={{ flex: 1, backgroundColor: isDark ? '#1A1A1A' : '#F8F8F8' }}>
      <Tabs
        tabBar={(props) => <CustomTabBar {...props} />}
        screenOptions={{
          animation: 'shift',
          headerShown: false,
        }}
      >
        <Tabs.Screen
          name="translate"
          options={{ title: 'Translate', headerShown: false }}
        /> 
        <Tabs.Screen
          name="learn"
          options={{ title: 'Learn', headerShown: false }}
        />
                 <Tabs.Screen
          name="quiz"
          options={{ title: 'Quiz', headerShown: false }}
        />
        <Tabs.Screen
          name="profile"
          options={{ title: 'Profile', headerShown: false }}
        />
      </Tabs>
    </View>
  );
};

export default _layout;