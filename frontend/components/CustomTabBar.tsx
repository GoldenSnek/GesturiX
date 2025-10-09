import React, { useEffect, useRef } from 'react';
import { View, TouchableOpacity, Text, Image, Animated, Dimensions, Platform } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';

// Define the type for our tab names
type TabName = 'translate' | 'compose' | 'learn' | 'profile';

const { width } = Dimensions.get('window'); // Get screen width for positioning

export default function CustomTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();
  
  const translateX = useRef(new Animated.Value(0)).current; 
  const tabItemWidth = useRef(width / state.routes.length).current;

  useEffect(() => {
    const toValue = state.index * tabItemWidth;

    Animated.spring(translateX, {
      toValue,
      stiffness: 1000,
      damping: 50,
      mass: 1,
      useNativeDriver: true,
    }).start();
  }, [state.index, translateX, tabItemWidth]);

  return (
    <View
      // Main floating container
      className={`absolute bottom-0 left-0 right-0 items-center`}
      style={{
        paddingBottom: insets.bottom + (Platform.OS === 'android' ? 0 : 20), 
      }}
    >
      <View
        // The oval-shaped container for the tabs
        // 👈 COLOR & TRANSPARENCY: Using darkbg/90 (90% opacity) for semi-transparency
        className="flex-row items-center justify-around bg-darkbg/80 rounded-full px-2 py-2 shadow-xl"
        style={{
          width: width * 0.9, 
          height: 65,
          // Using a dark shadow to match the dark bar background
          shadowColor: '#1a1a1a', 
          shadowOffset: { width: 0, height: 6 },
          shadowOpacity: 0.9,
          shadowRadius: 10,
          elevation: 15,
        }}
      >
        {state.routes.map((route, index) => {
          const { options } = descriptors[route.key];
          const label = options.title || route.name;
          const isFocused = state.index === index;

          const iconSource = {
            'translate': require('../assets/images/Translate-icon.png'),
            'compose': require('../assets/images/Compose-icon.png'),
            'learn': require('../assets/images/Learn-icon.png'),
            'profile': require('../assets/images/Profile-icon.png'),
          }[route.name as TabName] || null;

          const handlePress = () => {
            const event = navigation.emit({
              type: 'tabPress',
              target: route.key,
              canPreventDefault: true,
            });

            if (!isFocused && !event.defaultPrevented) {
              navigation.navigate(route.name, route.params);
            }
          };

          // Determine color based on focus state using custom classes
          const iconTintColor = isFocused ? '#FF6B00' : '#F8F8F8'; // Direct colors for tintStyle
          const textClassName = isFocused ? 'text-accent' : 'text-secondary';
          
          return (
            <TouchableOpacity
              key={route.key}
              accessibilityRole="button"
              accessibilityState={isFocused ? { selected: true } : {}}
              accessibilityLabel={options.tabBarAccessibilityLabel}
              onPress={handlePress}
              className={`flex-1 justify-center items-center h-full ${isFocused ? 'py-1' : ''}`}
              style={{ width: tabItemWidth }}
            >
              {isFocused ? (
                // FOCUSED: Show Icon (Accent) and Text (Accent)
                <View className="flex-col items-center justify-center">
                  <Image 
                    source={iconSource} 
                    className="w-6 h-6" 
                    // Use the direct hex code for tintColor as Nativewind can't directly use classes here
                    style={{ tintColor: iconTintColor }} 
                  />
                  <Text 
                    className={`text-xs font-semibold mt-1 ${textClassName}`} // text-accent
                    numberOfLines={1}
                  >
                    {label}
                  </Text>
                </View>
              ) : (
                // UNFOCUSED: Show ONLY Icon (Secondary)
                <Image 
                  source={iconSource} 
                  className="w-6 h-6" 
                  // Use the direct hex code for tintColor
                  style={{ tintColor: iconTintColor }} 
                />
              )}
            </TouchableOpacity>
          );
        })}
      </View>
    </View>
  );
}