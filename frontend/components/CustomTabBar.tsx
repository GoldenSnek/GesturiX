import React, { useEffect, useRef } from 'react';
import { View, TouchableOpacity, Text, Image, Animated } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { BottomTabBarProps } from '@react-navigation/bottom-tabs';

// Define the type for our tab names to prevent implicit 'any' type errors
type TabName = 'translate' | 'compose' | 'learn' | 'profile';

export default function CustomTabBar({ state, descriptors, navigation }: BottomTabBarProps) {
  const insets = useSafeAreaInsets();

  return (
    <View 
      className="flex-row bg- border-t border-gray-200" 
      style={{ paddingBottom: insets.bottom }}
    >
      {state.routes.map((route, index) => {
        const { options } = descriptors[route.key];
        // Safely get the label, preferring the title, otherwise the route name
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

        return (
          <TouchableOpacity
            key={route.key}
            accessibilityRole="button"
            accessibilityState={isFocused ? { selected: true } : {}}
            accessibilityLabel={options.tabBarAccessibilityLabel}
            onPress={handlePress}
            className="flex-1 justify-center items-center py-2"
          >
            <AnimatedIcon
              source={iconSource}
              isFocused={isFocused}
            />
            <Text
              className={`text-xs font-semibold mt-1 ${isFocused ? 'text-accent' : 'text-neutral'}`}
            >
              {label}
            </Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
}

// Define the props for the animated icon component
interface AnimatedIconProps {
  source: any;
  isFocused: boolean;
}

const AnimatedIcon = ({ source, isFocused }: AnimatedIconProps) => {
    const scaleAnim = useRef(new Animated.Value(1)).current;

    useEffect(() => {
        if (isFocused) {
            Animated.timing(scaleAnim, {
                toValue: 1.2,
                duration: 200,
                useNativeDriver: true,
            }).start();
        } else {
            Animated.timing(scaleAnim, {
                toValue: 1,
                duration: 200,
                useNativeDriver: true,
            }).start();
        }
    }, [isFocused]);

    return (
        <Animated.Image
            source={source}
            className="w-6 h-6"
            style={{ transform: [{ scale: scaleAnim }] }}
        />
    );
};