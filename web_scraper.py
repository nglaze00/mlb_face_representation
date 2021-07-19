from selenium import webdriver
import requests


def get_player_info(self, mlb_players_url):
    """
    gets player IDs and names from MLB players page

    :param mlb_players_url: url to MLB players page
    :returns: lists of MLB player ids and names
    """
    # prepare the option for the chrome driver
    options = webdriver.ChromeOptions()
    options.add_argument('headless')

    # start chrome browser
    browser = webdriver.Chrome(options=options)

    ids, names = [], []

    browser.get(mlb_players_url)

    index_element = browser.find_element_by_id('players-index')

    for player in index_element.find_element_by_css_selector('*').find_elements_by_css_selector('*'):
        link = player.get_attribute('href')
        if link is None:
            continue

        splits = link.split('/')
        if len(splits) != 5:
            continue

        ids.append(splits[-1].split('-')[-1])
        names.append(' '.join(splits[-1].split('-')[:-1]))

    browser.quit()
    return ids, names


def scrape_headshot(generic_headshot_url, id, local_path):
    """
    Gets the headshot for a given player id, and saves it to the given file path

    :param generic_headshot_url: url that headshots are stored at, with a format variable for the id
    :param local_path: path to save image to
    :return: True if saved successfully, False if failed
    """

    response = requests.get(generic_headshot_url.format(id))

    if not response.ok:
        print("response")
        return False

    with open(local_path, 'wb') as handle:
        for block in response.iter_content(1024):
            if not block:
                break

            handle.write(block)

    return True

def scrape_all_headshots(ids, names, generic_headshot_url, local_folder):
    """
    Scrapes all headshots using ids and saves them as their player name in local_folder
    :param ids: list of player ids
    :param names: list of player names
    :param generic_headshot_url: URL to headshot images, with format variable for id
    :param local_folder: folder to save images in
    """
    for i, n in zip(ids, names):
        if not scrape_headshot(generic_headshot_url, i, local_folder + '/{}.jpg'.format(n)):
            print('Error for ' + n)


