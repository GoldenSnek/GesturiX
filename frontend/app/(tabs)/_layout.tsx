import { View, Text } from 'react-native'
import React from 'react'
import {Tabs} from "expo-router";
import CustomTabBar from '../../components/CustomTabBar';

const _layout = () => {
  return (
    <Tabs
      tabBar={props => <CustomTabBar {...props} />}
      screenOptions={{
        headerShown: false,
      }}
    >
        <Tabs.Screen
        name="translate"
        options={{ title: 'Translate', headerShown: false}}
        />
        <Tabs.Screen
        name="compose"
        options={{ title: 'Compose', headerShown: false}}
        />
        <Tabs.Screen
        name="learn"
        options={{ title: 'Learn', headerShown: false}}
        />
        <Tabs.Screen
        name="profile"
        options={{ title: 'Profile', headerShown: false}}
        />
    </Tabs>
  ) 
}

export default _layout