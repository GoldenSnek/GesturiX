import * as Linking from 'expo-linking';

export function redirectSystemPath({ path }: { path: string }) {
  console.log("=================================================");
  console.log(`[NATIVE INTENT INTERCEPTED]`);
  console.log(`Original Path: ${path}`);

  if (path.includes('/--/oauth-callback')) {
    // using the correct Linking.parse just to satisfy any internal checks
    Linking.parse(path); 
    
    //Extract everything after the '#' or '?'
    const fragment = path.split('#')[1] || path.split('?')[1];

    // The file is app/(stack)/OAuthCallback.tsx, so the path is /OAuthCallback
    const targetPath = fragment 
        ? `/OAuthCallback?${fragment}` 
        : '/OAuthCallback';

    console.log(`Rewriting Path to: ${targetPath}`);
    console.log("=================================================");
    
    // This will now point to your actual file at app/(stack)/OAuthCallback.tsx
    return targetPath;
  }
  
  console.log(`Path accepted: ${path}`);
  console.log("=================================================");
  
  return path;
}